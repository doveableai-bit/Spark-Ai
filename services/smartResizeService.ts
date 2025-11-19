
import { GoogleGenAI, Modality } from "@google/genai";

const API_KEY = process.env.API_KEY;
if (!API_KEY) {
  throw new Error("API_KEY environment variable not set");
}
const ai = new GoogleGenAI({ apiKey: API_KEY });

export interface SmartResizeRequest {
  originalImage: string;        // Base64 without prefix
  targetAspectRatio: string;
  resizeMode: 'extend' | 'crop' | 'smart';  // 'extend' for background extension
  maintainElements: {
    face: boolean;
    dress: boolean;
    pose: boolean;
    environment: boolean;
    style: boolean;
  };
}

export interface SmartResizeResult {
  resizedImage: string;
  originalRatio: string;
  newRatio: string;
  resizeMethod: string;
  extendedBackground: boolean;
}

export class SmartResizeAI {
  
  async smartResizeImage(request: SmartResizeRequest): Promise<SmartResizeResult> {
    try {
      const { originalImage, targetAspectRatio, resizeMode, maintainElements } = request;
      
      const prompt = this.buildSmartResizePrompt(targetAspectRatio, resizeMode, maintainElements);
      
      const imagePart = { inlineData: { data: originalImage, mimeType: "image/png" } };
      const textPart = { text: prompt };

      const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash-image',
        contents: { parts: [imagePart, textPart] },
        config: {
            responseModalities: [Modality.IMAGE],
        }
      });

      let b64Image: string | null = null;

      for (const part of response.candidates?.[0]?.content?.parts || []) {
          if (part.inlineData) {
              b64Image = `data:${part.inlineData.mimeType};base64,${part.inlineData.data}`;
              break;
          }
      }
      
      if (!b64Image) throw new Error("Failed to generate resized image.");
      
      return {
        resizedImage: b64Image,
        originalRatio: 'original',
        newRatio: targetAspectRatio,
        resizeMethod: resizeMode,
        extendedBackground: resizeMode === 'extend'
      };

    } catch (error: any) {
      console.error("Smart resize error:", error);
      const msg = error.message || error.toString() || '';
      if (msg.includes('429') || msg.includes('RESOURCE_EXHAUSTED')) {
          throw new Error("Quota limit reached. Please try again later.");
      }
      throw new Error("Failed to resize image with smart background extension");
    }
  }

  private buildSmartResizePrompt(
    targetRatio: string,
    resizeMode: string,
    maintainElements: any
  ): string {
    let prompt = `CRITICAL TASK: CREATE A NEW IMAGE with Aspect Ratio ${targetRatio}.\n\n`;
    
    prompt += `TARGET: ${targetRatio} (Ignore input image dimensions)\n`;
    prompt += `MODE: ${resizeMode.toUpperCase()} / OUTPAINTING\n\n`;
    
    prompt += "CONSISTENCY CHECKLIST (MUST PRESERVE):\n";
    
    if (maintainElements.face) {
      prompt += `✅ FACE: Exact identity, features, expression\n`;
    }
    
    if (maintainElements.dress) {
      prompt += `✅ DRESS: Exact outfit, colors, patterns\n`;
    }
    
    if (maintainElements.pose) {
      prompt += `✅ POSE: Same body position\n`;
    }
    
    if (maintainElements.environment) {
      prompt += `✅ ENVIRONMENT: Same scene context\n`;
    }
    
    if (maintainElements.style) {
      prompt += `✅ STYLE: Same artistic style\n`;
    }

    prompt += `\nEXECUTION INSTRUCTIONS:\n`;
    prompt += `1. START A NEW CANVAS with aspect ratio ${targetRatio}.\n`;
    prompt += `2. PLACE the subject from the reference image into this new canvas.\n`;
    prompt += `3. EXTEND (Outpaint) the background to fill the remaining space seamlessly.\n`;
    prompt += `4. DO NOT stretch the subject. Keep proportions correct.\n`;
    prompt += `5. Ensure the result is a single, full-bleed image.\n`;
    
    prompt += `\nOUTPUT FORMAT:\n`;
    prompt += `- Final Image MUST have aspect ratio ${targetRatio}.\n`;
    prompt += `- NO black bars, NO frames, NO UI elements.\n`;
    prompt += `- High Quality, seamless extension.\n`;

    return prompt;
  }

  // Analyze image to understand background for smart extension
  async analyzeImageBackground(imageBase64: string): Promise<{
    backgroundType: 'solid' | 'textured' | 'scene' | 'gradient' | 'pattern';
    dominantColors: string[];
    hasHorizon: boolean;
    backgroundElements: string[];
    extensionSuggestions: string[];
  }> {
    try {
      const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash',
        contents: {
            parts: [
                { inlineData: { mimeType: "image/png", data: imageBase64 } },
                { text: `
                Analyze this image's background and provide details for smart extension:
                
                1. Background Type: solid, textured, scene, gradient, or pattern
                2. Dominant Colors: List main background colors
                3. Has Horizon: true/false
                4. Background Elements: What objects/elements are in background?
                5. Extension Suggestions: How to best extend this background?
                
                Respond ONLY with a valid JSON object:
                {
                  "backgroundType": "string",
                  "dominantColors": ["color1", "color2"],
                  "hasHorizon": boolean,
                  "backgroundElements": ["element1", "element2"],
                  "extensionSuggestions": ["suggestion1", "suggestion2"]
                }
                ` }
            ]
        }
      });

      const text = response.text || "{}";
      const jsonStr = text.replace(/```json/g, '').replace(/```/g, '').trim();
      return JSON.parse(jsonStr);

    } catch (error) {
      console.error("Background analysis error:", error);
      return {
        backgroundType: 'scene',
        dominantColors: ['#ffffff'],
        hasHorizon: false,
        backgroundElements: [],
        extensionSuggestions: ['Extend background naturally']
      };
    }
  }

  getAspectRatios() {
    return [
      { value: '1:1', label: '1:1 Square', icon: '⬜', description: 'Instagram Posts' },
      { value: '4:5', label: '4:5 Portrait', icon: '📱', description: 'Instagram Portrait' },
      { value: '9:16', label: '9:16 Story', icon: '📲', description: 'Stories, Reels' },
      { value: '16:9', label: '16:9 Landscape', icon: '🏞️', description: 'Desktop, TV' },
      { value: '3:2', label: '3:2 Classic', icon: '📸', description: 'Photography' },
      { value: '2:3', label: '2:3 Vertical', icon: '🖼️', description: 'Portrait Photos' },
      { value: '21:9', label: '21:9 Ultra Wide', icon: '🎬', description: 'Cinematic' },
      { value: '9:21', label: '9:21 Ultra Tall', icon: '📳', description: 'Tall Mobile' }
    ];
  }

  getResizeModes() {
    return [
      { 
        value: 'extend', 
        label: 'Smart Extend', 
        icon: '🔍', 
        description: 'Extend background intelligently (Recommended)',
        bestFor: 'All image types'
      },
      { 
        value: 'crop', 
        label: 'Smart Crop', 
        icon: '✂️', 
        description: 'Crop image intelligently',
        bestFor: 'Images with extra space'
      },
      { 
        value: 'smart', 
        label: 'Auto Smart', 
        icon: '🤖', 
        description: 'AI decides best method',
        bestFor: 'Automatic processing'
      }
    ];
  }
}
