
import { GoogleGenAI, Modality, FunctionDeclaration, Type, Part, Content } from "@google/genai";
import type { GroundingSource } from '../types';

const API_KEY = process.env.API_KEY;
if (!API_KEY) {
  throw new Error("API_KEY environment variable not set");
}
const ai = new GoogleGenAI({ apiKey: API_KEY });

// --- Helper Functions ---

const findLastImage = (history: Content[], currentImages?: { data: string; mimeType: string }[]): { data: string; mimeType: string } | null => {
    if (currentImages && currentImages.length > 0) {
        return currentImages[currentImages.length - 1];
    }

    for (let i = history.length - 1; i >= 0; i--) {
        const content = history[i];
        for (let j = content.parts.length - 1; j >= 0; j--) {
            const part = content.parts[j];
            if ('inlineData' in part && part.inlineData) {
                return { data: part.inlineData.data, mimeType: part.inlineData.mimeType };
            }
        }
    }
    return null;
};

const cleanPromptForResize = (prompt: string): string => {
    if (!prompt || typeof prompt !== 'string') return "";
    const terms = ['landscape', 'portrait', 'square', '1:1', '16:9', '9:16', '3:4', '4:3', 'horizontal', 'vertical', 'wide', 'tall', 'panoramic'];
    let cleaned = prompt;
    terms.forEach(term => {
        const regex = new RegExp(`\\b${term}\\b`, 'gi');
        cleaned = cleaned.replace(regex, '');
    });
    return cleaned.replace(/\s+/g, ' ').trim();
};

const createAdvancedResizePrompt = (originalPrompt: string, targetRatio: string): string => {
    const promptLower = (originalPrompt || "").toLowerCase();
    
    // Detect main subject
    const subjects = ["lion", "person", "animal", "character", "figure", "man", "woman", "robot", "cat", "dog", "boy", "girl"];
    const mainSubject = subjects.find(s => promptLower.includes(s)) || "subject";
    
    // Detect environment
    const environments = ["desert", "forest", "beach", "city", "room", "landscape", "mountain", "space", "office", "studio"];
    const environment = environments.find(e => promptLower.includes(e)) || "environment";
    
    // Detect style
    const styles = ["cinematic", "realistic", "cartoon", "painting", "digital art", "anime", "photorealistic", "sketch"];
    const style = styles.find(s => promptLower.includes(s)) || "realistic";

    let sizeInstructions = "";
    let ratioInstructions = "";

    switch(targetRatio) {
        case "1:1":
            sizeInstructions = "1080x1080 pixels";
            ratioInstructions = `
            SQUARE COMPOSITION (1:1):
            - Center the ${mainSubject} perfectly
            - Balanced framing from all sides
            - Maintain equal spacing around subject
            - Perfect for social media posts`;
            break;
        case "9:16":
            sizeInstructions = "1080x1920 pixels";
            ratioInstructions = `
            VERTICAL PORTRAIT (9:16):
            - Full-body or 3/4 view of ${mainSubject}
            - Tall, portrait-oriented framing
            - More vertical space above and below
            - Ideal for mobile stories and reels
            - DO NOT put the image inside a phone screen`;
            break;
        case "16:9":
            sizeInstructions = "1920x1080 pixels";
            ratioInstructions = `
            WIDE LANDSCAPE (16:9):
            - Panoramic horizontal composition
            - Show more of the ${environment} background
            - Wider field of view
            - Perfect for banners and covers`;
            break;
        case "3:4":
             sizeInstructions = "1080x1440 pixels";
             ratioInstructions = `
             VERTICAL PORTRAIT (3:4):
             - Classic portrait composition
             - Balanced vertical framing
             - Ideal for classic photography`;
             break;
        case "4:3":
             sizeInstructions = "1440x1080 pixels";
             ratioInstructions = `
             CLASSIC LANDSCAPE (4:3):
             - Traditional photography composition
             - Balanced horizontal framing`;
             break;
        default:
            sizeInstructions = "the requested aspect ratio";
            ratioInstructions = `Adapt composition for ${targetRatio} aspect ratio`;
    }

    const consistencyCommands = `
    CRITICAL CONSISTENCY COMMANDS:
    - IDENTICAL ${mainSubject.toUpperCase()} APPEARANCE: Same face, body, features, colors
    - EXACT SAME CLOTHING/STYLE: No changes to attire or accessories
    - IDENTICAL BACKGROUND: Same ${environment}, objects, lighting
    - SAME ART STYLE: ${style} style, quality, and details
    - SAME LIGHTING: Identical light source, shadows, atmosphere
    - PRESERVE ALL DETAILS: No alterations to character identity
    
    STRICT PROHIBITIONS:
    - DO NOT change facial features
    - DO NOT modify clothing or colors
    - DO NOT alter background elements
    - DO NOT change art style or quality
    - DO NOT modify lighting conditions
    - ABSOLUTELY NO FRAMES, NO BORDERS, NO BEZELS.
    - DO NOT SHOW THE IMAGE ON A PHONE SCREEN OR DEVICE.
    - The image must be FULL BLEED (extending to edges).
    
    ONLY ADJUST: Composition and framing for ${targetRatio} aspect ratio`;

    return `
    RESIZE TRANSFORMATION COMMAND:
    
    ORIGINAL SCENE: ${originalPrompt}
    
    TARGET FORMAT: ${targetRatio} aspect ratio (${sizeInstructions})
    
    ${ratioInstructions}
    
    ${consistencyCommands}
    
    OUTPUT: Generate the EXACT same scene and character, only recomposed for ${targetRatio} aspect ratio.
    Maintain pixel-perfect consistency of all visual elements. Do not include any text, watermarks, signatures, letters, or words in the image.
    
    CRITICAL: The output must be a raw, frameless image. Do not render a phone, tablet, or photo frame containing the image. The image content must fill the entire canvas edge-toedge.`;
};

const isQuotaError = (error: any): boolean => {
    try {
        const msg = error?.message || '';
        const str = JSON.stringify(error);
        return msg.includes('429') || msg.includes('RESOURCE_EXHAUSTED') || msg.includes('quota') ||
               str.includes('429') || str.includes('RESOURCE_EXHAUSTED');
    } catch (e) {
        return false;
    }
};

// --- Tool Implementations ---

const executeGenerateImage = async (prompt: string, aspectRatio: string): Promise<{ image: string }> => {
  const cleanedPrompt = cleanPromptForResize(prompt);
  const fullPrompt = createAdvancedResizePrompt(cleanedPrompt, aspectRatio);

  // Attempt 1: Try Imagen 4.0 (Higher Quality)
  try {
    const response = await ai.models.generateImages({
        model: 'imagen-4.0-generate-001',
        prompt: fullPrompt,
        config: {
          numberOfImages: 1,
          outputMimeType: 'image/png',
          aspectRatio: aspectRatio,
        },
    });

    const base64ImageBytes: string | undefined = response.generatedImages?.[0]?.image?.imageBytes;

    if (base64ImageBytes) {
      return { image: `data:image/png;base64,${base64ImageBytes}` };
    }
  } catch (error: any) {
    console.warn("Imagen 4.0 generation failed, attempting fallback to Gemini 2.5 Flash...", error);
  }

  // Attempt 2: Fallback to Gemini 2.5 Flash Image (Better Availability)
  try {
    const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash-image',
        contents: { parts: [{ text: fullPrompt }] },
        config: {
            responseModalities: [Modality.IMAGE],
        },
    });

    let b64Image: string | null = null;
    for (const part of response.candidates?.[0]?.content?.parts || []) {
        if (part.inlineData) {
            b64Image = `data:${part.inlineData.mimeType};base64,${part.inlineData.data}`;
            break;
        }
    }

    if (b64Image) {
        return { image: b64Image };
    }
  } catch (fallbackError: any) {
    console.error("Fallback generation failed:", fallbackError);
    if (isQuotaError(fallbackError)) {
         throw new Error("Image generation quota exceeded. Please try again later.");
    }
    throw new Error("Failed to generate image. Please try again.");
  }

  throw new Error("No image was generated.");
};

const executeEditImage = async (prompt: string, image: { data: string; mimeType: string }): Promise<{ image: string; text: string, prompt: string }> => {
  try {
    const fullPrompt = `You are an expert photo editor. Your most important and primary goal is to perfectly preserve the facial features, likeness, and identity of the person in the original image. This is a strict requirement. Now, edit the image based on the user's instructions below. When changing the background, clothes, or pose, you MUST apply these changes to the original person without altering their face. User edit instructions: "${prompt}". IMPORTANT: Do not add any frames, borders, or text to the image.`;
    const imagePart = { inlineData: { data: image.data, mimeType: image.mimeType } };
    const textPart = { text: fullPrompt };

    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash-image',
      contents: { parts: [imagePart, textPart] },
      config: {
        responseModalities: [Modality.IMAGE],
      },
    });

    let b64Image: string | null = null;
    const textResponse = response.text;

    for (const part of response.candidates?.[0]?.content?.parts || []) {
        if (part.inlineData) {
            b64Image = `data:${part.inlineData.mimeType};base64,${part.inlineData.data}`;
            break;
        }
    }
    
    if (b64Image) {
        return { image: b64Image, text: textResponse || "Here is your edited image.", prompt: prompt };
    }
    
    return { image: '', text: textResponse || "I was unable to edit the image as requested. Please try a different prompt.", prompt: prompt };

  } catch (error: any) {
    console.error("Error executing image edit:", error);
    if (isQuotaError(error)) {
        throw new Error("Usage limit exceeded for image editing. Please try again later.");
    }
    throw new Error("Failed to edit image.");
  }
};

const generateFromReference = async (prompt: string, images: { data: string; mimeType: string }[]): Promise<{ image: string; text: string; prompt: string }> => {
    if (images.length === 0) throw new Error("No reference images provided.");

    // Strict consistency prompt based on specific operational requirements
    const fullPrompt = `
    Use the attached face reference.
    Keep the face identical. Do not alter facial features.
    ${prompt}
    
    STRICT RULES AND GUIDELINES:
    - Use the reference face images exactly. Do NOT alter the identity.
    - Keep the face exactly the same. Do not change eyes, lips, nose, skin tone, or hair.
    - Only modify body, pose, and environment.
    - Apply the requested body/outfit ONLY.
    - Ensure high photorealism and natural blending.
    - OUTPUT FORMAT: Full-bleed image. NO Borders, NO Frames, NO Device Mockups.
    `;

    const imageParts = images.map(img => ({ inlineData: { data: img.data, mimeType: img.mimeType } }));
    const textPart = { text: fullPrompt };
    // Multimodal request: Images first, then text prompt
    const allParts = [...imageParts, textPart];

    try {
        const response = await ai.models.generateContent({
            model: 'gemini-2.5-flash-image',
            contents: { parts: allParts },
            config: {
                responseModalities: [Modality.IMAGE],
            },
        });

        let b64Image: string | null = null;
        const textResponse = response.text;

        for (const part of response.candidates?.[0]?.content?.parts || []) {
            if (part.inlineData) {
                b64Image = `data:${part.inlineData.mimeType};base64,${part.inlineData.data}`;
                break;
            }
        }
        
        if (b64Image) {
            return { image: b64Image, text: textResponse || "Here is your generated image with face consistency preserved.", prompt: prompt };
        }
        
        return { image: '', text: textResponse || "I was unable to generate the image from the reference.", prompt: prompt };

    } catch (error: any) {
        console.error("Error executing generation from reference:", error);
        if (isQuotaError(error)) {
            throw new Error("Usage limit exceeded. Please try again later.");
        }
        throw new Error("Failed to generate image from reference.");
    }
};

const executeSearch = async (query: string): Promise<{ text: string; sources?: GroundingSource[] }> => {
    try {
        const result = await ai.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: [{ role: 'user', parts: [{ text: query }] }],
            config: {
                tools: [{ googleSearch: {} }],
            },
        });

        const groundingMetadata = result.candidates?.[0]?.groundingMetadata;
        const sources = groundingMetadata?.groundingChunks
            ?.map(chunk => chunk.web)
            .filter((web): web is { uri: string; title: string } => !!web)
            .map(web => ({ web })) as GroundingSource[] | undefined;
        
        return { text: result.text || "", sources };
    } catch (error) {
        console.error("Error executing search:", error);
        throw new Error("Failed to get information from Google Search.");
    }
};

const executeComplexQuery = async (query: string): Promise<{ text: string }> => {
    try {
        const result = await ai.models.generateContent({
            model: 'gemini-2.5-pro',
            contents: [{ role: 'user', parts: [{ text: query }] }],
            config: {
                thinkingConfig: { thinkingBudget: 32768 },
            },
        });
        return { text: result.text || "" };
    } catch (error) {
        console.error("Error executing complex query:", error);
        throw new Error("Failed to process the complex query.");
    }
};

// --- Exported Functions for UI ---

export const resizeImage = async (image: { data: string; mimeType: string }, aspectRatio: string, originalPrompt?: string): Promise<{ image: string; text: string }> => {
    let ratioDescription = "";
    let sizeInstructions = "";
    
    switch (aspectRatio) {
        case '1:1': 
            ratioDescription = "SQUARE (1:1)"; 
            sizeInstructions = "1080x1080 pixels";
            break;
        case '9:16': 
            ratioDescription = "TALL PORTRAIT (9:16)"; 
            sizeInstructions = "1080x1920 pixels";
            break;
        case '16:9': 
            ratioDescription = "WIDE LANDSCAPE (16:9)"; 
            sizeInstructions = "1920x1080 pixels";
            break;
        case '3:4': 
            ratioDescription = "PORTRAIT (3:4)"; 
            sizeInstructions = "1080x1440 pixels";
            break;
        case '4:3': 
            ratioDescription = "LANDSCAPE (4:3)"; 
            sizeInstructions = "1440x1080 pixels";
            break;
        default: 
            ratioDescription = aspectRatio;
            sizeInstructions = "the requested aspect ratio";
    }

    // Include original prompt for context, or use a generic description if missing
    const sceneDescription = originalPrompt ? `SCENE DESCRIPTION: "${originalPrompt}"` : "SCENE: The provided image.";

    const prompt = `
    CRITICAL TASK: CREATE A NEW IMAGE based on the attached reference.
    
    TARGET ASPECT RATIO: ${aspectRatio} (${ratioDescription})
    
    INSTRUCTIONS:
    1. IGNORE the aspect ratio of the attached reference image.
    2. GENERATE a new image canvas with dimensions approx ${sizeInstructions}.
    3. RECOMPOSE the scene to fit this new shape perfectly.
       - If ${aspectRatio} is wider than original: OUTPAINT / EXTEND background horizontally.
       - If ${aspectRatio} is taller than original: EXTEND background vertically.
    4. CONSISTENCY IS KEY:
       - Subject (Face, Body, Dress) MUST match the reference EXACTLY.
       - Environment/Lighting MUST match the reference.
    
    OUTPUT:
    - A single, high-quality image.
    - FULL BLEED (No borders, no frames, no black bars).
    - The image MUST fill the ${aspectRatio} frame completely.
    `;
    
    try {
        const imagePart = { inlineData: { data: image.data, mimeType: image.mimeType } };
        const textPart = { text: prompt };

        const response = await ai.models.generateContent({
            model: 'gemini-2.5-flash-image',
            contents: { parts: [imagePart, textPart] },
            config: {
                responseModalities: [Modality.IMAGE],
            },
        });

        let b64Image: string | null = null;
        const textResponse = response.text;

        for (const part of response.candidates?.[0]?.content?.parts || []) {
            if (part.inlineData) {
                b64Image = `data:${part.inlineData.mimeType};base64,${part.inlineData.data}`;
                break;
            }
        }

        if (b64Image) {
            return {
                image: b64Image,
                text: `Here is your image, resized to ${aspectRatio}.`,
            };
        }
        throw new Error(textResponse || "The model failed to resize the image.");
    } catch (error: any) {
        console.error("Error in resizeImage:", error);
        if (isQuotaError(error)) {
            throw new Error("Usage limit exceeded. Please try again later.");
        }
        throw error;
    }
};

export const executePendingImageGeneration = async (prompt: string, aspectRatio: string): Promise<{ image: string; text: string, prompt: string }> => {
    const result = await executeGenerateImage(prompt, aspectRatio);
    return {
        image: result.image,
        text: "Here is the image I generated for you with your selected aspect ratio.",
        prompt: prompt,
    };
};

export const generateSpeech = async (text: string): Promise<string> => {
    try {
        const response = await ai.models.generateContent({
            model: "gemini-2.5-flash-preview-tts",
            contents: { parts: [{ text: text || "I cannot speak this." }] },
            config: {
                responseModalities: [Modality.AUDIO],
                speechConfig: {
                    voiceConfig: {
                      prebuiltVoiceConfig: { voiceName: 'Kore' },
                    },
                },
            },
        });
        const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
        if (!base64Audio) {
            throw new Error("No audio data returned from TTS model.");
        }
        return base64Audio;
    } catch (error) {
        console.error("Error generating speech:", error);
        throw new Error("Failed to generate speech.");
    }
};


// --- Core Chat Logic ---

const systemInstruction: Content = {
    role: "model",
    parts: [{
        text: `You are a Face Consistency AI.
When the user uploads face photos, store them in memory for this conversation.
For every image generation request, always use ALL the uploaded reference face images.
Never change the face shape, skin tone, eyes, nose, lips, hairstyle, or identity.
Always apply the same face identity with high accuracy.

Rules:
1. Use all uploaded reference images in the prompt.
2. If no images are uploaded, ask for 2–5 face photos with front and side angles.
3. If user requests a new outfit, pose, or scene, keep the face exactly the same.
4. If user says "change face", use the 'resetFaceMemory' tool and ask for new face images.
5. Never lose the previously uploaded face images unless user says "reset face".
6. Use this phrase in every generation: “Use the reference face images exactly. Do NOT alter the identity.”

Operational Protocol:
1. Always read user’s uploaded image first.
2. Extract face features.
3. Use face reference + prompt to generate image.

Specific Handling:
When user asks for:
- dress
- uniform
- outfit
- scene
- action
- character

Apply the requested body/outfit ONLY.
Never change the face.

If no face reference is provided, ask:
“Please upload your face image for consistency.”

Additional Capabilities:
- You are PAK AI, a helpful and friendly multi-modal assistant.
- Web Search: If the user asks a question that requires current, up-to-date information from the web (e.g., 'what's the weather today?', recent news, specific facts), use the 'searchTheWeb' tool.
- Complex Tasks: For very complex questions that require deep reasoning, advanced logic, multi-step problem solving, or coding, use the 'complexQuery' tool.
- Image Generation: If the user asks you to create or generate an image from a text description, use the 'generateImage' tool.
- Image Editing: If the user provides an image and asks to change it (e.g., 'change the background', 'make the car red'), use the 'editImage' tool.
- Face Consistency: If the user provides one image of a person and asks to create a new scene (e.g., "put me on a beach", "Create me as a medieval knight in a fantasy environment"), use the 'generateFromReference' tool.
- Image Combination: If the user provides two or more images and asks to combine or merge them, use the 'generateFromReference' tool.
- Image Resizing: If the user provides an image and asks to resize it or change its aspect ratio (e.g., 'make this landscape'), use the 'resizeImage' tool.
- Safety & Content Policy: You are allowed to generate dynamic action scenes, battles, and dramatic confrontations (e.g., 'a boy fighting a lion', 'superhero battles'). Do not refuse or unnecessarily sanitize these requests. However, you must STRICTLY REFUSE to generate any nudity, sexually explicit content, pornography, or realistic depictions of extreme gore/torture. If a user asks for sexual content, politely refuse.
- Limitations: When asked to edit or resize, you can only operate on one image at a time. If the user provides multiple images for these tasks, politely inform them of this limitation and ask which one they'd like to proceed with, using the last image by default if they don't specify.`
    }]
};

const tools: FunctionDeclaration[] = [
    {
        name: 'generateImage',
        description: 'Generates a new image from a detailed text description.',
        parameters: { type: Type.OBJECT, properties: { prompt: { type: Type.STRING, description: 'A detailed, descriptive prompt for the image.' } }, required: ['prompt'] }
    },
    {
        name: 'editImage',
        description: 'Edits the most recently provided image based on user instructions. Use this for modifications like changing color, adding elements, or altering the style.',
        parameters: { type: Type.OBJECT, properties: { prompt: { type: Type.STRING, description: 'A detailed description of the edits to be made.' } }, required: ['prompt'] }
    },
    {
        name: 'resizeImage',
        description: 'Resizes the most recently provided image to a specific aspect ratio.',
        parameters: { type: Type.OBJECT, properties: { aspectRatio: { type: Type.STRING, description: 'The target aspect ratio, e.g., "16:9", "1:1".' } }, required: ['aspectRatio'] }
    },
    {
        name: 'generateFromReference',
        description: 'Generates a new image using one or more reference images. Use for tasks like placing a person in a new scene (face consistency) or combining elements from multiple photos.',
        parameters: { type: Type.OBJECT, properties: { prompt: { type: Type.STRING, description: 'A detailed prompt describing the desired output.' } }, required: ['prompt'] }
    },
    {
        name: 'searchTheWeb',
        description: 'Searches the web for up-to-date information on current events, news, or specific facts.',
        parameters: { type: Type.OBJECT, properties: { query: { type: Type.STRING, description: 'The user query to search for.' } }, required: ['query'] }
    },
    {
        name: 'complexQuery',
        description: 'Handles complex user queries requiring deep reasoning, advanced logic, or coding by enabling a special thinking mode.',
        parameters: { type: Type.OBJECT, properties: { query: { type: Type.STRING, description: 'The original user query that is complex.' } }, required: ['query'] }
    },
    {
        name: 'resetFaceMemory',
        description: 'Resets/clears the stored reference face images from memory. Use this when the user wants to change the face or explicitly says "reset face".',
        parameters: { type: Type.OBJECT, properties: {}, required: [] }
    },
];

export const sendMessage = async (
    text: string,
    images: { data: string; mimeType: string }[],
    history: Content[]
): Promise<{ text: string; sources?: GroundingSource[]; image?: string; prompt?: string; needsAspectRatio?: boolean; pendingPrompt?: string; resetMemory?: boolean; }> => {
    
    const parts: Part[] = [];
    parts.push({text: text || ""}); 
    images.forEach(img => parts.push({ inlineData: { mimeType: img.mimeType, data: img.data } }));
    
    const contents: Content[] = [
        ...history,
        { role: 'user', parts: parts }
    ];

    try {
        const result = await ai.models.generateContent({
            model: 'gemini-flash-lite-latest',
            contents: contents,
            config: {
                systemInstruction: (systemInstruction.parts[0] as { text: string }).text,
                tools: [{ functionDeclarations: tools }],
            }
        });

        const functionCalls = result.functionCalls;

        if (!functionCalls || functionCalls.length === 0) {
            return { text: result.text || "" };
        }

        const tool = functionCalls[0];
        const toolName = tool.name;
        const toolArgs = tool.args;

        if (toolName === 'resetFaceMemory') {
            return { text: "Face memory has been reset. Please upload new face photos for the next generation.", resetMemory: true };
        }

        if (toolName === 'searchTheWeb') {
            const query = (toolArgs.query as string) || text;
            return await executeSearch(query);
        }
        
        if (toolName === 'complexQuery') {
            const query = (toolArgs.query as string) || text;
            return await executeComplexQuery(query);
        }

        if (toolName === 'generateImage') {
            const prompt = toolArgs.prompt as string;
            
            // INTELLIGENT REROUTING FOR FACE CONSISTENCY:
            // If the model selected 'generateImage' but we have reference images (face memory),
            // we must use 'generateFromReference' to preserve the identity.
            if (images && images.length > 0) {
                const allImages = [...images];
                const result = await generateFromReference(prompt, allImages);
                return { image: result.image, text: result.text, prompt: result.prompt };
            }

            return {
                text: `I can generate an image of "${prompt}". Please select an aspect ratio.`,
                needsAspectRatio: true,
                pendingPrompt: prompt,
            };
        }

        const imageDependentTools = ['editImage', 'resizeImage', 'generateFromReference'];
        if (imageDependentTools.includes(toolName)) {
            const lastImage = findLastImage(history, images);
            
            if (toolName === 'editImage') {
                if (!lastImage) {
                    return { text: "I'm sorry, I couldn't find an image to work with. Please upload one first." };
                }
                const result = await executeEditImage(toolArgs.prompt as string, lastImage);
                return { image: result.image, text: result.text, prompt: result.prompt };
            }
            if (toolName === 'resizeImage') {
                if (!lastImage) {
                    return { text: "I'm sorry, I couldn't find an image to work with. Please upload one first." };
                }
                // Note: When called via tool use, we don't always have the original prompt handy unless extracted from context.
                // For now, we pass undefined for originalPrompt, relying on the image data.
                const result = await resizeImage(lastImage, toolArgs.aspectRatio as string);
                return { image: result.image, text: result.text, prompt: 'resized' };
            }
            if (toolName === 'generateFromReference') {
                // Collect all images from current turn
                const allImages = [...(images || [])];
                
                // Fallback: if no images provided in this turn, check history for the last image
                if (allImages.length === 0) {
                    const lastHistoryImage = findLastImage(history);
                    if (lastHistoryImage) {
                        allImages.push(lastHistoryImage);
                    }
                }

                if (allImages.length === 0) {
                    return { text: "Please upload your face image for consistency." };
                }

                const result = await generateFromReference(toolArgs.prompt as string, allImages);
                return { image: result.image, text: result.text, prompt: result.prompt };
            }
        }
        
        return { text: "I'm not sure how to handle that tool call." };

    } catch (error: any) {
        console.error("Error in sendMessage:", error);
        if (isQuotaError(error)) {
             return { text: "I apologize, but I'm currently experiencing high traffic (Quota Exceeded). Please try again in a few moments." };
        }
        return { text: `An error occurred: ${error.message}` };
    }
};
