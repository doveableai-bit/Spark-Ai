
import { GoogleGenAI, Modality } from "@google/genai";

const API_KEY = process.env.API_KEY;
if (!API_KEY) {
  throw new Error("API_KEY environment variable not set");
}
const ai = new GoogleGenAI({ apiKey: API_KEY });

export interface ConsistencySettings {
  face: 'consistent' | 'change';
  dress: 'consistent' | 'change'; 
  background: 'consistent' | 'change';
  environment: 'consistent' | 'change';
  currentFaceBase64?: string;
  currentDressBase64?: string;
  currentBackgroundBase64?: string;
  currentEnvironmentBase64?: string;
}

export interface GenerationResult {
  imageData: string;
  consistencySettings: ConsistencySettings;
  generatedElements: {
    face: boolean;
    dress: boolean;
    background: boolean;
    environment: boolean;
  };
}

export class CompleteConsistencyAI {
  private faceMemory: string | null = null;
  private dressMemory: string | null = null;
  private backgroundMemory: string | null = null;
  private environmentMemory: string | null = null;

  // Set reference images (expects base64 string without data URI header)
  setFaceReference(base64: string) {
    this.faceMemory = base64;
  }

  setDressReference(base64: string) {
    this.dressMemory = base64;
  }

  setBackgroundReference(base64: string) {
    this.backgroundMemory = base64;
  }

  setEnvironmentReference(base64: string) {
    this.environmentMemory = base64;
  }

  // Generate with complete control
  async generateWithControl(
    userPrompt: string,
    settings: ConsistencySettings,
    aspectRatio: string = "1:1"
  ): Promise<GenerationResult> {
    try {
      const prompt = this.buildSmartPrompt(userPrompt, settings, aspectRatio);
      const parts: any[] = [];

      // Add reference images based on consistency settings
      if (settings.face === 'consistent' && this.faceMemory) {
        parts.push({
          inlineData: {
            mimeType: "image/png",
            data: this.faceMemory
          }
        });
      }

      if (settings.dress === 'consistent' && this.dressMemory) {
        parts.push({
          inlineData: {
            mimeType: "image/png", 
            data: this.dressMemory
          }
        });
      }

      if (settings.background === 'consistent' && this.backgroundMemory) {
        parts.push({
          inlineData: {
            mimeType: "image/png",
            data: this.backgroundMemory
          }
        });
      }

      if (settings.environment === 'consistent' && this.environmentMemory) {
        parts.push({
          inlineData: {
            mimeType: "image/png",
            data: this.environmentMemory
          }
        });
      }

      // Add text prompt
      parts.push({ text: prompt });

      const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash-image',
        contents: { parts },
        config: {
            responseModalities: [Modality.IMAGE],
        }
      });

      let imageData = '';
      for (const part of response.candidates?.[0]?.content?.parts || []) {
        if (part.inlineData) {
            imageData = part.inlineData.data;
            break;
        }
      }
      
      if (!imageData) throw new Error("No image generated");

      return {
        imageData: `data:image/png;base64,${imageData}`,
        consistencySettings: settings,
        generatedElements: {
          face: settings.face === 'change',
          dress: settings.dress === 'change', 
          background: settings.background === 'change',
          environment: settings.environment === 'change'
        }
      };

    } catch (error: any) {
      console.error("Generation error:", error);
      const msg = error.message || error.toString() || '';
      const str = JSON.stringify(error);
      if (msg.includes('429') || msg.includes('RESOURCE_EXHAUSTED') || str.includes('429') || str.includes('RESOURCE_EXHAUSTED')) {
        throw new Error("Quota limit reached. Please try again later.");
      }
      throw new Error("Failed to generate image with consistency control");
    }
  }

  private buildSmartPrompt(userPrompt: string, settings: ConsistencySettings, aspectRatio: string): string {
    let prompt = `USER REQUEST: "${userPrompt}"\n\n`;
    
    prompt += "CRITICAL CONSISTENCY INSTRUCTIONS:\n";

    if (settings.face === 'consistent') {
      prompt += "🔷 FACE: Keep EXACTLY same face from reference - identical features, structure, identity\n";
    } else {
      prompt += "🔷 FACE: Create new face based on context\n";
    }

    if (settings.dress === 'consistent') {
      prompt += "👗 DRESS: Maintain EXACT same clothing/outfit from reference\n";
    } else {
      prompt += "👗 DRESS: Create new clothing based on user request\n";
    }

    if (settings.background === 'consistent') {
      prompt += "🏞️ BACKGROUND: Keep EXACT same background scene from reference\n";
    } else {
      prompt += "🏞️ BACKGROUND: Create new background based on user request\n";
    }

    if (settings.environment === 'consistent') {
      prompt += "🌍 ENVIRONMENT: Maintain EXACT same environment, lighting, atmosphere from reference\n";
    } else {
      prompt += "🌍 ENVIRONMENT: Create new environment based on user request\n";
    }

    prompt += `
ADDITIONAL REQUIREMENTS:
- No borders, frames, or white spaces
- Full bleed image, aspect ratio: ${aspectRatio}
- High quality, photorealistic
- Seamless integration of all elements
- Follow user prompt for any new elements
- OUTPUT: A single high-quality image.
    `;

    return prompt;
  }

  // Extract specific elements from generated image
  async extractElement(
    imageBase64: string, 
    elementType: 'face' | 'dress' | 'background' | 'environment'
  ): Promise<string> {
    try {
      const extractionPrompt = this.getElementExtractionPrompt(elementType);
      
      const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash-image',
        contents: {
            parts: [
                { inlineData: { mimeType: "image/png", data: imageBase64 } },
                { text: extractionPrompt }
            ]
        },
        config: {
            responseModalities: [Modality.IMAGE]
        }
      });

      let extractedData = '';
      for (const part of response.candidates?.[0]?.content?.parts || []) {
        if (part.inlineData) {
            extractedData = part.inlineData.data;
            break;
        }
      }
      
      if (!extractedData) throw new Error(`Failed to extract ${elementType}`);

      return extractedData;

    } catch (error: any) {
      console.error(`Extraction error for ${elementType}:`, error);
      const msg = error.message || error.toString() || '';
      const str = JSON.stringify(error);
      if (msg.includes('429') || msg.includes('RESOURCE_EXHAUSTED') || str.includes('429') || str.includes('RESOURCE_EXHAUSTED')) {
        throw new Error("Quota limit reached. Please try again later.");
      }
      throw new Error(`Failed to extract ${elementType}`);
    }
  }

  private getElementExtractionPrompt(elementType: string): string {
    const prompts = {
      face: "Extract only the face with clear features. Remove everything else. Provide clean face reference.",
      dress: "Extract only the clothing/outfit. Remove face and background. Provide clean dress reference.", 
      background: "Extract only the background scene. Remove person/subject. Provide clean background reference.",
      environment: "Extract the environment with lighting and atmosphere. Remove main subject. Provide environment reference."
    };

    return prompts[elementType as keyof typeof prompts];
  }

  // Get current memories
  getMemories() {
    return {
      face: this.faceMemory,
      dress: this.dressMemory,
      background: this.backgroundMemory, 
      environment: this.environmentMemory
    };
  }

  // Clear specific memory
  clearMemory(element: 'face' | 'dress' | 'background' | 'environment') {
    switch(element) {
      case 'face': this.faceMemory = null; break;
      case 'dress': this.dressMemory = null; break;
      case 'background': this.backgroundMemory = null; break;
      case 'environment': this.environmentMemory = null; break;
    }
  }

  // Clear all memories
  clearAllMemories() {
    this.faceMemory = null;
    this.dressMemory = null;
    this.backgroundMemory = null;
    this.environmentMemory = null;
  }
}
