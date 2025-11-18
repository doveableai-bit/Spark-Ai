
export enum Sender {
  User = 'user',
  Bot = 'bot',
}

export interface GroundingSource {
  web: {
    uri: string;
    title: string;
  };
}

export interface ChatMessage {
  id: string;
  sender: Sender;
  text: string;
  images?: string[]; // Array of base64 image data URLs
  sources?: GroundingSource[];
  isLoading?: boolean;
  isError?: boolean;
  prompt?: string; // The prompt used for image generation/editing
  needsAspectRatio?: boolean; // If true, UI should show aspect ratio selector
}