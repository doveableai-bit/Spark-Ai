
import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { Sender } from './types';
import type { ChatMessage } from './types';
import { sendMessage, executePendingImageGeneration, resizeImage, generateSpeech } from './services/geminiService';
import { CompleteConsistencyAI, ConsistencySettings, GenerationResult } from './services/completeConsistencyService';
import { Content } from '@google/genai';

// --- Constants ---
const MAX_CHAR_LIMIT = 8000;
const ASPECT_RATIOS = [
    { ratio: '1:1', label: 'Square', description: 'For Instagram Posts' },
    { ratio: '9:16', label: 'Portrait', description: 'For Stories & Reels' },
    { ratio: '16:9', label: 'Landscape', description: 'For Thumbnails & Banners' },
    { ratio: '3:4', label: 'Portrait (3:4)', description: 'Classic Portrait' },
    { ratio: '4:3', label: 'Landscape (4:3)', description: 'Classic Landscape' },
];

// --- Helper Functions ---
const fileToBase64 = (file: File): Promise<string> =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve((reader.result as string).split(',')[1]);
    reader.onerror = (error) => reject(error);
  });

const decode = (base64: string): Uint8Array => {
    const binaryString = atob(base64);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes;
};

async function decodeAudioData(
  data: Uint8Array,
  ctx: AudioContext,
  sampleRate: number,
  numChannels: number,
): Promise<AudioBuffer> {
  const dataInt16 = new Int16Array(data.buffer);
  const frameCount = dataInt16.length / numChannels;
  const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);

  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < frameCount; i++) {
      channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
    }
  }
  return buffer;
}

// --- Icons ---
const SparklesIcon = () => (<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5"><path fillRule="evenodd" d="M10.868 2.884c.321-.772 1.415-.772 1.736 0l1.291 3.118c.261.63.852 1.054 1.523 1.12l3.434.5c1.13.164 1.588 1.555.744 2.368l-2.484 2.42c-.469.458-.682 1.12-.549 1.758l.586 3.42a1.363 1.363 0 0 1-1.98 1.432L10 16.45l-3.056 1.606a1.363 1.363 0 0 1-1.98-1.432l.586-3.42a1.99 1.99 0 0 0-.55-1.758l-2.484-2.42c-.844-.813-.386-2.204.744-2.368l3.434-.5c.671-.066 1.262-.49 1.523-1.12l1.29-3.118Z" clipRule="evenodd" /></svg>);
const UserIcon = () => (<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5"><path fillRule="evenodd" d="M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0Zm-5.5-2.5a2.5 2.5 0 1 1-5 0 2.5 2.5 0 0 1 5 0ZM10 12a5.99 5.99 0 0 0-4.793 2.39A6.483 6.483 0 0 0 10 16.5a6.483 6.483 0 0 0 4.793-2.11A5.99 5.99 0 0 0 10 12Z" clipRule="evenodd" /></svg>);
const PaperAirplaneIcon = () => (<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5"><path d="M3.105 2.289a.75.75 0 0 0-.826.95l1.414 4.949a.75.75 0 0 0 .95.54l3.118-1.559a.75.75 0 0 1 .527.955l-2.093 4.185a.75.75 0 0 0 .504.986l5.05 1.443a.75.75 0 0 0 .986-.504l4.185-10.462a.75.75 0 0 0-.527-.955L4.413 3.457a.75.75 0 0 0-.95.54l-1.414-1.718Z" /><path d="M12.91 8.351a.75.75 0 0 0-1.06 0l-2.093 2.093a.75.75 0 0 0 1.06 1.06L12.91 9.41a.75.75 0 0 0 0-1.06Z" /></svg>);
const PaperClipIcon = () => (<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5"><path fillRule="evenodd" d="M15.621 4.379a3 3 0 0 0-4.242 0l-7 7a3 3 0 0 0 4.241 4.243h.001l.497-.5a.75.75 0 0 1 1.06 1.06l-.497.5a4.5 4.5 0 1 1-6.364-6.364l7-7a4.5 4.5 0 0 1 6.364 6.364l-3.182 3.182a.75.75 0 0 0 1.061 1.06l3.182-3.182a3 3 0 0 0 0-4.242Z" clipRule="evenodd" /></svg>);
const XCircleIcon = () => (<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5"><path fillRule="evenodd" d="M10 18a8 8 0 1 0 0-16 8 8 0 0 0 0 16Zm3.707-10.707a1 1 0 0 0-1.414-1.414L10 8.586 7.707 6.293a1 1 0 0 0-1.414 1.414L8.586 10l-2.293 2.293a1 1 0 1 0 1.414 1.414L10 11.414l2.293 2.293a1 1 0 0 0 1.414-1.414L11.414 10l2.293-2.293Z" clipRule="evenodd" /></svg>);
const MicrophoneIcon = () => (<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5"><path d="M7 4a3 3 0 0 1 6 0v6a3 3 0 1 1-6 0V4Z" /><path d="M5.5 8.5a.5.5 0 0 1 .5.5v1.5a.5.5 0 0 1-1 0V9a.5.5 0 0 1 .5-.5Z" /><path d="M10 18a.5.5 0 0 0 .5-.5v-2.29a6.474 6.474 0 0 0 5-6.21v-1.5a.5.5 0 0 0-1 0v1.5a5.474 5.474 0 0 1-10 0v-1.5a.5.5 0 0 0-1 0v1.5a6.474 6.474 0 0 0 5 6.21v2.29a.5.5 0 0 0 .5.5Z" /></svg>);
const SpeakerWaveIcon = () => (<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5"><path strokeLinecap="round" strokeLinejoin="round" d="M19.114 5.636a9 9 0 0 1 0 12.728M16.463 8.288a5.25 5.25 0 0 1 0 7.424M6.75 8.25l4.72-4.72a.75.75 0 0 1 1.28.53v15.88a.75.75 0 0 1-1.28.53l-4.72-4.72H4.51c-.88 0-1.704-.507-1.938-1.354A9.01 9.01 0 0 1 2.25 12c0-.83.112-1.633.322-2.396C2.806 8.756 3.63 8.25 4.51 8.25H6.75Z" /></svg>);
const SpeakerXMarkIcon = () => (<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5"><path strokeLinecap="round" strokeLinejoin="round" d="M17.25 9.75 19.5 12m0 0 2.25 2.25M19.5 12l2.25-2.25M19.5 12l-2.25 2.25m-10.5-6.375a9 9 0 0 1 12.728 0M12.75 6.035A5.25 5.25 0 0 1 18 10.262m-10.5 0A5.25 5.25 0 0 1 12 6.035m-7.5 4.227 4.72-4.72a.75.75 0 0 1 1.28.53v15.88a.75.75 0 0 1-1.28.53l-4.72-4.72H4.51c-.88 0-1.704-.507-1.938-1.354A9.01 9.01 0 0 1 2.25 12c0-.83.112-1.633.322-2.396C2.806 8.756 3.63 8.25 4.51 8.25H6.75Z" /></svg>);
const ResizeIcon = () => (<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5"><path d="M10 3a.75.75 0 0 1 .75.75v1.5h1.5a.75.75 0 0 1 0 1.5h-1.5v1.5a.75.75 0 0 1-1.5 0v-1.5h-1.5a.75.75 0 0 1 0-1.5h1.5v-1.5A.75.75 0 0 1 10 3ZM10 17a.75.75 0 0 1-.75-.75v-1.5h-1.5a.75.75 0 0 1 0-1.5h1.5v-1.5a.75.75 0 0 1 1.5 0v1.5h1.5a.75.75 0 0 1 0 1.5h-1.5v1.5A.75.75 0 0 1 10 17ZM17 10a.75.75 0 0 1-.75.75h-1.5v1.5a.75.75 0 0 1-1.5 0v-1.5h-1.5a.75.75 0 0 1 0-1.5h1.5v-1.5a.75.75 0 0 1 1.5 0v1.5h1.5A.75.75 0 0 1 17 10ZM3 10a.75.75 0 0 1 .75-.75h1.5v-1.5a.75.75 0 0 1 1.5 0v1.5h1.5a.75.75 0 0 1 0 1.5h-1.5v1.5a.75.75 0 0 1-1.5 0v-1.5h-1.5A.75.75 0 0 1 3 10Z" /></svg>);
const ReuseIcon = () => (<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5"><path fillRule="evenodd" d="M15.312 11.424a5.5 5.5 0 0 1-9.201-4.42 5.5 5.5 0 0 1 10.852 2.132.75.75 0 0 1-1.423.433A4 4 0 0 0 6.13 8.355a4 4 0 0 0 7.938 1.48.75.75 0 0 1 1.244.789ZM4.688 8.576a.75.75 0 0 1 1.423-.433 4 4 0 0 0 8.239-1.523.75.75 0 0 1 1.244-.789 5.5 5.5 0 0 1-10.852-2.132.75.75 0 0 1 1.423-.433A4 4 0 0 0 13.87 11.645a4 4 0 0 0-7.938-1.48.75.75 0 0 1-1.244-.789Z" clipRule="evenodd" /></svg>);
const CopyIcon = () => (<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5"><path d="M7 3.5A1.5 1.5 0 0 1 8.5 2h3.879a1.5 1.5 0 0 1 1.06.44l3.122 3.121A1.5 1.5 0 0 1 17 6.621V16.5a1.5 1.5 0 0 1-1.5 1.5h-7A1.5 1.5 0 0 1 7 16.5v-13Z" /><path d="M4.5 6A1.5 1.5 0 0 0 3 7.5v10A1.5 1.5 0 0 0 4.5 19h7a1.5 1.5 0 0 0 1.5-1.5v-2a.75.75 0 0 0-1.5 0v2A.5.5 0 0 1 11.5 18h-7a.5.5 0 0 1-.5-.5v-10a.5.5 0 0 1 .5-.5h2a.75.75 0 0 0 0-1.5h-2Z" /></svg>);
const CopyResponseIcon = () => (<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5"><path strokeLinecap="round" strokeLinejoin="round" d="M15.75 17.25v3.375c0 .621-.504 1.125-1.125 1.125h-9.75c-.621 0-1.125-.504-1.125-1.125V7.875c0-.621.504-1.125 1.125-1.125H6.75a9.06 9.06 0 0 1 1.5.124m7.5 10.376h3.375c.621 0 1.125-.504 1.125-1.125V11.25c0-4.46-3.243-8.161-7.5-8.876a9.06 9.06 0 0 0-1.5-.124H9.375c-.621 0-1.125.504-1.125 1.125v3.5m7.5 10.375H9.375a1.125 1.125 0 0 1-1.125-1.125v-9.25m12 3.375-3.375-3.375m0 0L18 5.25m-3.375 3.375a1.125 1.125 0 0 1 1.125-1.125h3.375c.621 0 1.125.504 1.125 1.125v9.75c0 .621-.504 1.125-1.125 1.125h-9.75a1.125 1.125 0 0 1-1.125-1.125V7.875c0-.621.504-1.125 1.125-1.125H6.75" /></svg>);
const LikeIcon = () => (<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5"><path strokeLinecap="round" strokeLinejoin="round" d="M6.633 10.5c.806 0 1.533-.446 2.031-1.08a9.041 9.041 0 0 1 2.861-2.4c.723-.384 1.35-.956 1.653-1.715a4.498 4.498 0 0 0 .322-1.672V3a.75.75 0 0 1 .75-.75A2.25 2.25 0 0 1 16.5 4.5c0 1.152-.26 2.243-.723 3.218-.266.558.107 1.282.725 1.282h3.126c1.026 0 1.945.694 2.054 1.715.045.422.068.85.068 1.285a11.95 11.95 0 0 1-2.649 7.521c-.388.482-.987.729-1.605.729H13.48c-.483 0-.964-.078-1.423-.23l-3.114-1.04a4.501 4.501 0 0 0-1.423-.23H5.904M6.633 10.5l-1.822-1.822a.75.75 0 0 0-1.06 0l-1.06 1.06a.75.75 0 0 0 0 1.06l1.06 1.06a.75.75 0 0 0 1.06 0l1.822-1.822Z" /></svg>);
const DislikeIcon = () => (<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5"><path strokeLinecap="round" strokeLinejoin="round" d="M17.367 13.5c-.806 0-1.533.446-2.031 1.08a9.041 9.041 0 0 1-2.861 2.4c-.723.384-1.35.956-1.653 1.715a4.498 4.498 0 0 0-.322 1.672v.75a.75.75 0 0 1-.75.75A2.25 2.25 0 0 1 7.5 19.5c0-1.152.26-2.243.723-3.218.266-.558-.107-1.282-.725-1.282H4.374c-1.026 0-1.945-.694-2.054-1.715A12.134 12.134 0 0 1 2.25 12c0-1.285.253-2.524.721-3.682.11-.26.311-.49.56-.653H10.52c.483 0 .964.078 1.423.23l3.114 1.04a4.501 4.501 0 0 0 1.423.23h1.994M17.367 13.5l1.822 1.822a.75.75 0 0 0 1.06 0l1.06-1.06a.75.75 0 0 0 0-1.06l-1.06-1.06a.75.75 0 0 0-1.06 0l-1.822 1.822Z" /></svg>);
const RegenerateIcon = () => (<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5"><path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 11.667 0 8.25 8.25 0 0 0 0-11.667l-3.182-3.182m0 0h-4.992m4.992 0v4.992" /></svg>);
const ShareIcon = () => (<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5"><path strokeLinecap="round" strokeLinejoin="round" d="M7.217 10.907a2.25 2.25 0 1 0 0 2.186m0-2.186c.18.324.283.696.283 1.093s-.103.77-.283 1.093m0-2.186 9.566-5.314m-9.566 7.5 9.566 5.314m0 0a2.25 2.25 0 1 0 3.935 2.186 2.25 2.25 0 0 0-3.935-2.186Zm0-12.814a2.25 2.25 0 1 0 3.933-2.186 2.25 2.25 0 0 0-3.933 2.186Z" /></svg>);
const DownloadIcon = () => (<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5"><path d="M10.75 2.75a.75.75 0 0 0-1.5 0v8.614L6.295 8.235a.75.75 0 1 0-1.09 1.03l4.25 4.5a.75.75 0 0 0 1.09 0l4.25-4.5a.75.75 0 0 0-1.09-1.03l-2.955 3.129V2.75Z" /><path d="M3.5 12.75a.75.75 0 0 0-1.5 0v2.5A2.75 2.75 0 0 0 4.75 18h10.5A2.75 2.75 0 0 0 18 15.25v-2.5a.75.75 0 0 0-1.5 0v2.5c0 .69-.56 1.25-1.25 1.25H4.75c-.69 0-1.25-.56-1.25-1.25v-2.5Z" /></svg>);
const AddUserIcon = () => (<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5"><path d="M11 5a3 3 0 1 1-6 0 3 3 0 0 1 6 0ZM2.615 16.428a1.224 1.224 0 0 1-.569-1.175 6.002 6.002 0 0 1 11.908 0c.058.467-.172.92-.57 1.174A9.953 9.953 0 0 1 7 18a9.953 9.953 0 0 1-4.385-1.572Z" /><path d="M16.25 5.75a.75.75 0 0 0-1.5 0v2h-2a.75.75 0 0 0 0 1.5h2v2a.75.75 0 0 0 1.5 0v-2h2a.75.75 0 0 0 0-1.5h-2v-2Z" /></svg>);
const DashboardIcon = () => (<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5"><path d="M2 4.25A2.25 2.25 0 0 1 4.25 2h2.5A2.25 2.25 0 0 1 9 4.25v2.5A2.25 2.25 0 0 1 6.75 9h-2.5A2.25 2.25 0 0 1 2 6.75v-2.5ZM2 13.25A2.25 2.25 0 0 1 4.25 11h2.5A2.25 2.25 0 0 1 9 13.25v2.5A2.25 2.25 0 0 1 6.75 18h-2.5A2.25 2.25 0 0 1 2 15.75v-2.5ZM11 4.25A2.25 2.25 0 0 1 13.25 2h2.5A2.25 2.25 0 0 1 18 4.25v2.5A2.25 2.25 0 0 1 15.75 9h-2.5A2.25 2.25 0 0 1 11 6.75v-2.5ZM11 13.25A2.25 2.25 0 0 1 13.25 11h2.5A2.25 2.25 0 0 1 18 13.25v2.5A2.25 2.25 0 0 1 15.75 18h-2.5A2.25 2.25 0 0 1 11 15.75v-2.5Z" /></svg>);
const ChatBubbleIcon = () => (<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5"><path fillRule="evenodd" d="M10 2c-2.236 0-4.43.18-6.57.524C1.993 2.755 1 4.014 1 5.426v5.148c0 1.413.993 2.67 2.43 2.902.848.137 1.705.248 2.57.331v3.443a.75.75 0 0 0 1.28.53l3.58-3.579a.78.78 0 0 1 .527-.224 41.202 41.202 0 0 0 5.183-.5c1.437-.232 2.43-1.49 2.43-2.903V5.426c0-1.413-.993-2.67-2.43-2.902A41.289 41.289 0 0 0 10 2Zm0 7a1 1 0 1 0 0-2 1 1 0 0 0 0 2ZM8 8a1 1 0 1 1-2 0 1 1 0 0 1 2 0Zm5 1a1 1 0 1 0 0-2 1 1 0 0 0 0 2Z" clipRule="evenodd" /></svg>);

// --- Child Components (Chat UI) ---
interface MessageProps {
    message: ChatMessage;
    promptToCopy?: string;
    isSpeaking: boolean;
    onResizeClick: (message: ChatMessage) => void;
    onReuseClick: (message: ChatMessage) => void;
    onCopyPromptClick: (prompt: string) => void;
    onCopyResponseClick: (text: string) => void;
    onDownloadClick: (message: ChatMessage) => void;
    onAspectRatioSelect: (messageId: string, prompt: string, ratio: string) => void;
    onLikeClick: (messageId: string) => void;
    onDislikeClick: (messageId: string) => void;
    onSpeakClick: (text: string, messageId: string) => void;
    onRegenerateClick: (messageId: string) => void;
    onShareClick: (text: string) => void;
}

const IconButton: React.FC<{ onClick: () => void; icon: React.ReactNode; label: string, active?: boolean, isLoading?: boolean }> = ({ onClick, icon, label, active = false, isLoading = false }) => (
    <button
        onClick={onClick}
        className={`p-1.5 rounded-md transition-colors ${active ? 'text-blue-400 bg-blue-500/20' : 'text-gray-400 hover:bg-gray-600 hover:text-gray-200'}`}
        aria-label={label}
        disabled={isLoading}
    >
        {isLoading ? <div className="w-5 h-5 border-2 border-gray-400 border-t-transparent rounded-full animate-spin"></div> : icon}
    </button>
);

const Message: React.FC<MessageProps> = (props) => {
    const { message, promptToCopy, isSpeaking, onResizeClick, onReuseClick, onCopyPromptClick, onCopyResponseClick, onDownloadClick, onAspectRatioSelect, onLikeClick, onDislikeClick, onSpeakClick, onRegenerateClick, onShareClick } = props;
    const isUser = message.sender === Sender.User;
    const Icon = isUser ? UserIcon : SparklesIcon;

    const parsedText = useMemo(() => {
        const displayText = (message.text && typeof message.text === 'string') ? message.text : "";
        if (!displayText || typeof (window as any).marked === 'undefined') {
            return { __html: displayText.replace(/\n/g, '<br />') };
        }
        return { __html: (window as any).marked.parse(displayText) };
    }, [message.text]);

    const showImageActions = !isUser && message.images && message.images.length > 0;
    const showActionBar = !isUser && !message.isLoading && !message.needsAspectRatio && message.id !== 'init';

    return (
        <div className={`flex items-start gap-4 my-4 ${isUser ? 'justify-end' : ''}`}>
            {!isUser && (
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-blue-400 to-indigo-600 flex items-center justify-center text-white">
                    <Icon />
                </div>
            )}
            <div className={`max-w-xl p-4 rounded-2xl shadow-md ${isUser ? 'bg-blue-600 text-white rounded-br-none' : 'bg-gray-700 text-gray-200 rounded-bl-none'}`}>
                {message.isLoading ? (
                    <div className="flex items-center space-x-2">
                        <div className="w-2 h-2 bg-gray-300 rounded-full animate-pulse"></div>
                        <div className="w-2 h-2 bg-gray-300 rounded-full animate-pulse [animation-delay:0.2s]"></div>
                        <div className="w-2 h-2 bg-gray-300 rounded-full animate-pulse [animation-delay:0.4s]"></div>
                    </div>
                ) : (
                    <>
                        <div className="prose prose-invert prose-sm" dangerouslySetInnerHTML={parsedText} />
                        
                        {message.prompt && !message.needsAspectRatio && (
                             <div className="mt-3 pt-2 border-t border-dashed border-gray-600">
                                <p className="text-xs text-gray-400 font-mono whitespace-pre-wrap">{message.prompt}</p>
                            </div>
                        )}

                        {message.images && message.images.length > 0 && (
                            <div className={`mt-3 grid gap-2 ${message.images.length > 1 ? 'grid-cols-2' : 'grid-cols-1'}`}>
                                {message.images.map((imgSrc, index) => (
                                    <img key={index} src={imgSrc} alt={`content ${index + 1}`} className="rounded-lg w-full" />
                                ))}
                            </div>
                        )}

                        {showActionBar && (
                            <div className="flex items-start flex-wrap gap-4 mt-3 pt-3 border-t border-gray-600">
                                { !showImageActions ? (
                                    <>
                                        <IconButton onClick={() => onCopyResponseClick(message.text)} icon={<CopyResponseIcon />} label="Copy Response" />
                                        <IconButton onClick={() => onLikeClick(message.id)} icon={<LikeIcon />} label="Like" active={message.feedback === 'liked'} />
                                        <IconButton onClick={() => onDislikeClick(message.id)} icon={<DislikeIcon />} label="Dislike" active={message.feedback === 'disliked'} />
                                        <IconButton onClick={() => onSpeakClick(message.text, message.id)} icon={<SpeakerWaveIcon />} label="Speak" isLoading={isSpeaking} />
                                        <IconButton onClick={() => onRegenerateClick(message.id)} icon={<RegenerateIcon />} label="Regenerate" />
                                        <IconButton onClick={() => onShareClick(message.text)} icon={<ShareIcon />} label="Share" />
                                        <div className="h-4 w-px bg-gray-600 mx-1 self-center"></div>
                                    </>
                                ) : (
                                    <>
                                        <div className="flex flex-col items-center text-center">
                                            <IconButton onClick={() => onResizeClick(message)} icon={<ResizeIcon />} label="Resize" />
                                            <span className="text-xs text-gray-400 mt-1">Resize</span>
                                        </div>
                                        <div className="flex flex-col items-center text-center">
                                            <IconButton onClick={() => onReuseClick(message)} icon={<ReuseIcon />} label="Reuse" />
                                            <span className="text-xs text-gray-400 mt-1">Reuse</span>
                                        </div>
                                        <div className="flex flex-col items-center text-center">
                                            <IconButton onClick={() => onDownloadClick(message)} icon={<DownloadIcon />} label="Download" />
                                            <span className="text-xs text-gray-400 mt-1">Download</span>
                                        </div>
                                        <div className="h-10 w-px bg-gray-600 self-center mx-1"></div>
                                    </>
                                )}

                                {promptToCopy && (
                                    showImageActions ? (
                                        <div className="flex flex-col items-center text-center">
                                            <IconButton onClick={() => onCopyPromptClick(promptToCopy)} icon={<CopyIcon />} label="Copy Prompt" />
                                            <span className="text-xs text-gray-400 mt-1 leading-tight">Copy Prompt</span>
                                        </div>
                                    ) : (
                                        <IconButton onClick={() => onCopyPromptClick(promptToCopy)} icon={<CopyIcon />} label="Copy Prompt" />
                                    )
                                )}
                            </div>
                        )}

                        {message.needsAspectRatio && message.prompt && (
                            <div className="mt-4 space-y-2">
                                {ASPECT_RATIOS.map(({ ratio, label, description }) => (
                                    <button
                                        key={ratio}
                                        onClick={() => onAspectRatioSelect(message.id, message.prompt!, ratio)}
                                        className="w-full text-left p-3 bg-gray-600/50 hover:bg-gray-600 rounded-lg transition-colors"
                                    >
                                        <p className="font-semibold text-white">{label} <span className="text-gray-400 font-normal">({ratio})</span></p>
                                        <p className="text-sm text-gray-400">{description}</p>
                                    </button>
                                ))}
                            </div>
                        )}
                    </>
                )}
            </div>
            {isUser && (
                 <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gray-600 flex items-center justify-center text-white">
                    <Icon />
                </div>
            )}
        </div>
    );
};

const Modal: React.FC<{ title: string; description?: string; onCancel: () => void; children: React.ReactNode }> = ({ title, description, onCancel, children }) => (
    <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 transition-opacity">
        <div className="bg-gray-800 rounded-2xl shadow-xl p-6 w-full max-w-md mx-4">
            <h2 className="text-xl font-bold text-white mb-2">{title}</h2>
            {description && <p className="text-gray-400 mb-6">{description}</p>}
            <div className="space-y-3">
                {children}
            </div>
            <button onClick={onCancel} className="w-full mt-6 p-2 bg-gray-600 hover:bg-gray-500 rounded-lg text-white font-semibold transition-colors">Cancel</button>
        </div>
    </div>
);

const Toast: React.FC<{ message: string }> = ({ message }) => (
    <div className="fixed bottom-24 left-1/2 -translate-x-1/2 bg-green-600 text-white px-4 py-2 rounded-lg shadow-lg animate-toast z-50">{message}</div>
);

// --- Consistency Studio Component ---
const ConsistencyStudio: React.FC<{ consistencyAI: CompleteConsistencyAI }> = ({ consistencyAI }) => {
    const [generatedImage, setGeneratedImage] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const [prompt, setPrompt] = useState('');
    const [aspectRatio, setAspectRatio] = useState('1:1');
    const [consistencySettings, setConsistencySettings] = useState<ConsistencySettings>({
        face: 'consistent', dress: 'change', background: 'change', environment: 'change'
    });
    const [currentUploadType, setCurrentUploadType] = useState<'face' | 'dress' | 'background' | 'environment'>('face');
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [memoryStatus, setMemoryStatus] = useState(consistencyAI.getMemories());
    const [toastMsg, setToastMsg] = useState<string|null>(null);

    const showToast = (msg: string) => { setToastMsg(msg); setTimeout(()=>setToastMsg(null), 2000); };

    const updateMemoryStatus = () => setMemoryStatus(consistencyAI.getMemories());

    const handleReferenceUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                const base64 = (e.target?.result as string).split(',')[1];
                switch(currentUploadType) {
                    case 'face': consistencyAI.setFaceReference(base64); break;
                    case 'dress': consistencyAI.setDressReference(base64); break;
                    case 'background': consistencyAI.setBackgroundReference(base64); break;
                    case 'environment': consistencyAI.setEnvironmentReference(base64); break;
                }
                updateMemoryStatus();
                showToast(`${currentUploadType.toUpperCase()} reference set!`);
            };
            reader.readAsDataURL(file);
        }
        if(event.target) event.target.value = '';
    };

    const triggerFileInput = (type: typeof currentUploadType) => {
        setCurrentUploadType(type);
        fileInputRef.current?.click();
    };

    const handleGenerate = async () => {
        if (!prompt.trim()) { showToast("Please enter a prompt"); return; }
        setLoading(true);
        try {
            const result = await consistencyAI.generateWithControl(prompt, consistencySettings, aspectRatio);
            setGeneratedImage(result.imageData);
        } catch (e: any) {
            console.error(e);
            showToast(e.message || "Generation failed.");
        } finally {
            setLoading(false);
        }
    };

    const handleExtract = async (type: typeof currentUploadType) => {
        if (!generatedImage) return;
        setLoading(true);
        try {
            const base64 = generatedImage.split(',')[1];
            const extracted = await consistencyAI.extractElement(base64, type);
            switch(type) {
                case 'face': consistencyAI.setFaceReference(extracted); setConsistencySettings(p=>({...p, face:'consistent'})); break;
                case 'dress': consistencyAI.setDressReference(extracted); setConsistencySettings(p=>({...p, dress:'consistent'})); break;
                case 'background': consistencyAI.setBackgroundReference(extracted); setConsistencySettings(p=>({...p, background:'consistent'})); break;
                case 'environment': consistencyAI.setEnvironmentReference(extracted); setConsistencySettings(p=>({...p, environment:'consistent'})); break;
            }
            updateMemoryStatus();
            showToast(`${type.toUpperCase()} extracted & set!`);
        } catch(e) { showToast(`Extraction failed for ${type}`); }
        finally { setLoading(false); }
    };

    const RefCard: React.FC<{ title: string, type: typeof currentUploadType, status: string | null }> = ({ title, type, status }) => (
        <div className="bg-gray-800 border border-gray-700 p-4 rounded-xl flex flex-col items-center gap-3 shadow-md hover:border-blue-500 transition-colors">
            <h3 className="font-bold text-gray-200">{title}</h3>
            <div className={`text-xs font-mono px-2 py-1 rounded ${status ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'}`}>
                {status ? "✅ SET" : "❌ EMPTY"}
            </div>
            {status && <img src={`data:image/png;base64,${status}`} className="w-16 h-16 object-cover rounded-lg border border-gray-600" alt="ref" />}
            <button onClick={() => triggerFileInput(type)} className="text-xs bg-blue-600 hover:bg-blue-700 text-white px-3 py-1.5 rounded transition-colors w-full">
                Upload
            </button>
        </div>
    );

    return (
        <div className="p-6 max-w-6xl mx-auto text-gray-200">
            {toastMsg && <Toast message={toastMsg} />}
            <input type="file" ref={fileInputRef} onChange={handleReferenceUpload} className="hidden" accept="image/*" />
            
            <div className="mb-8 text-center bg-gradient-to-r from-indigo-600 to-purple-600 p-10 rounded-3xl shadow-lg">
                <h2 className="text-4xl font-bold text-white mb-2">Consistency Studio</h2>
                <p className="text-indigo-100 text-lg">Granular control over Face, Dress, Background, and Environment.</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
                <RefCard title="👤 Face" type="face" status={memoryStatus.face} />
                <RefCard title="👗 Dress" type="dress" status={memoryStatus.dress} />
                <RefCard title="🏞️ Background" type="background" status={memoryStatus.background} />
                <RefCard title="🌍 Environment" type="environment" status={memoryStatus.environment} />
            </div>

            <div className="bg-gray-800 border border-gray-700 rounded-xl p-6 mb-8 shadow-xl">
                <h3 className="text-xl font-semibold text-gray-200 mb-6 border-b border-gray-700 pb-2 flex items-center gap-2">⚙️ Generation Settings</h3>
                
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
                    {(Object.keys(consistencySettings) as Array<keyof ConsistencySettings>).map(key => (
                        <div key={key} className="flex flex-col gap-2">
                            <label className="text-sm font-medium text-gray-400 capitalize tracking-wide">{key} Mode</label>
                            <select 
                                value={consistencySettings[key]} 
                                onChange={(e) => setConsistencySettings(p => ({...p, [key]: e.target.value}))}
                                className="bg-gray-900 border border-gray-600 text-white rounded-lg p-2.5 text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none transition-all"
                            >
                                <option value="consistent">Keep Consistent</option>
                                <option value="change">Generate New</option>
                            </select>
                        </div>
                    ))}
                </div>

                <div className="flex flex-col gap-3 mb-6">
                    <label className="text-sm font-medium text-gray-400">Prompt Description</label>
                    <textarea 
                        value={prompt} 
                        onChange={(e) => setPrompt(e.target.value)}
                        placeholder="Describe your scene in detail... (e.g. 'A cinematic shot of a futuristic city with neon lights')"
                        className="w-full bg-gray-900 text-white rounded-lg p-4 border border-gray-600 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none resize-none transition-all"
                        rows={3}
                    />
                </div>
                
                <div className="flex flex-wrap gap-4 items-center justify-between mt-4 bg-gray-900 p-4 rounded-lg">
                     <div className="flex items-center gap-3">
                        <label className="text-sm text-gray-400 font-medium">Aspect Ratio</label>
                        <select value={aspectRatio} onChange={(e) => setAspectRatio(e.target.value)} className="bg-gray-800 border border-gray-600 text-white rounded-lg p-2 text-sm focus:border-blue-500 outline-none cursor-pointer">
                            <option value="1:1">1:1 Square</option>
                            <option value="16:9">16:9 Landscape</option>
                            <option value="9:16">9:16 Portrait</option>
                        </select>
                     </div>
                     <button 
                        onClick={handleGenerate} 
                        disabled={loading}
                        className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-bold py-2.5 px-8 rounded-full shadow-lg disabled:opacity-50 disabled:cursor-not-allowed transform hover:-translate-y-0.5 transition-all"
                    >
                        {loading ? (
                            <span className="flex items-center gap-2">
                                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                                Generating...
                            </span>
                        ) : '✨ Generate Image'}
                    </button>
                </div>
            </div>

            {generatedImage && (
                <div className="bg-gray-800 border border-gray-700 rounded-xl p-8 text-center shadow-2xl animate-fade-in-up">
                     <h3 className="text-2xl font-semibold text-gray-200 mb-6">Generated Result</h3>
                     <div className="relative inline-block group">
                        <img src={generatedImage} alt="Generated" className="max-w-full max-h-[600px] mx-auto rounded-xl shadow-lg mb-8 border-2 border-gray-700" />
                        <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-10 transition-all rounded-xl pointer-events-none"></div>
                     </div>
                     
                     <div className="bg-gray-900 p-6 rounded-xl max-w-3xl mx-auto">
                        <h4 className="text-gray-400 text-sm font-semibold mb-4 uppercase tracking-wider">Extract Elements to References</h4>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                            <button onClick={() => handleExtract('face')} disabled={loading} className="bg-gray-800 hover:bg-blue-600 text-gray-300 hover:text-white py-2 px-4 rounded-lg border border-gray-700 transition-all text-sm font-medium">Extract Face</button>
                            <button onClick={() => handleExtract('dress')} disabled={loading} className="bg-gray-800 hover:bg-pink-600 text-gray-300 hover:text-white py-2 px-4 rounded-lg border border-gray-700 transition-all text-sm font-medium">Extract Dress</button>
                            <button onClick={() => handleExtract('background')} disabled={loading} className="bg-gray-800 hover:bg-green-600 text-gray-300 hover:text-white py-2 px-4 rounded-lg border border-gray-700 transition-all text-sm font-medium">Extract BG</button>
                            <button onClick={() => handleExtract('environment')} disabled={loading} className="bg-gray-800 hover:bg-purple-600 text-gray-300 hover:text-white py-2 px-4 rounded-lg border border-gray-700 transition-all text-sm font-medium">Extract Env</button>
                        </div>
                     </div>
                </div>
            )}
        </div>
    );
};


// --- Main App Component ---

type UploadedImage = { file: File, data: string, mimeType: string };

export default function App() {
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [input, setInput] = useState('');
    const [isProcessing, setIsProcessing] = useState(false);
    const [isRecording, setIsRecording] = useState(false);
    const [uploadedImages, setUploadedImages] = useState<UploadedImage[]>([]);
    const [faceReferences, setFaceReferences] = useState<UploadedImage[]>([]);
    const [isTtsEnabled, setIsTtsEnabled] = useState(false);
    const [resizeTask, setResizeTask] = useState<ChatMessage | null>(null);
    const [toastText, setToastText] = useState<string | null>(null);
    const [speakingMessageId, setSpeakingMessageId] = useState<string | null>(null);
    
    // New State for Mode Switching
    const [appMode, setAppMode] = useState<'chat' | 'studio'>('chat');
    const [consistencyAI] = useState(() => new CompleteConsistencyAI());

    const recognitionRef = useRef<any>(null);
    const messageListRef = useRef<HTMLDivElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const faceInputRef = useRef<HTMLInputElement>(null);
    const textareaRef = useRef<HTMLTextAreaElement>(null);
    const audioContextRef = useRef<AudioContext | null>(null);
    const audioSourceRef = useRef<AudioBufferSourceNode | null>(null);

    useEffect(() => {
        const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
        if (SpeechRecognition) {
            const recognition = new SpeechRecognition();
            recognition.continuous = true;
            recognition.interimResults = true;
            recognition.lang = 'en-US';

            recognition.onresult = (event: any) => {
                let interimTranscript = '';
                for (let i = event.resultIndex; i < event.results.length; ++i) {
                     if (event.results[i].isFinal) {
                        setInput(prev => prev + event.results[i][0].transcript);
                     } else {
                        interimTranscript += event.results[i][0].transcript;
                     }
                }
            };

            recognition.onstart = () => setIsRecording(true);
            recognition.onend = () => setIsRecording(false);
            recognition.onerror = (event: any) => {
                console.error("Speech recognition error", event.error);
                setIsRecording(false);
            };
            recognitionRef.current = recognition;
        }
    }, []);

    const cleanTextForSpeech = (text: string) => {
        if (!text || typeof text !== 'string') return "";
        return text
            .replace(/```[\s\S]*?```/g, '(Code block is displayed on screen.)')
            .replace(/\[([^\]]+)\]\([^\)]+\)/g, '$1')
            .replace(/(\*|_|`)/g, '');
    };
    
    useEffect(() => {
        if (!audioContextRef.current) {
            audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
        }
        return () => {
            if (audioSourceRef.current) {
                audioSourceRef.current.stop();
            }
        };
    }, []);

    const playAudio = useCallback(async (base64Audio: string, onEnded: () => void) => {
        if (!audioContextRef.current) return;

        if (audioSourceRef.current) {
            audioSourceRef.current.stop();
        }

        try {
            const audioBuffer = await decodeAudioData(
                decode(base64Audio),
                audioContextRef.current,
                24000, // Gemini TTS sample rate
                1, // mono
            );

            const source = audioContextRef.current.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(audioContextRef.current.destination);
            source.onended = () => {
                audioSourceRef.current = null;
                onEnded();
            };
            source.start();
            audioSourceRef.current = source;
        } catch (error) {
            console.error("Error playing audio:", error);
            onEnded(); // Clear loading state even on error
        }
    }, []);
    
    const handleSpeakClick = useCallback(async (text: string, messageId: string) => {
        if (!text || speakingMessageId === messageId) { // Prevent re-clicking while loading
            if (audioSourceRef.current) {
                 audioSourceRef.current.stop();
                 setSpeakingMessageId(null);
            }
            return;
        }
        
        setSpeakingMessageId(messageId);
        try {
            const cleanText = cleanTextForSpeech(text);
            const audioData = await generateSpeech(cleanText);
            await playAudio(audioData, () => setSpeakingMessageId(null));
        } catch (error) {
            console.error("Error in speak click handler:", error);
            showToast("Failed to generate speech.");
            setSpeakingMessageId(null);
        }

    }, [playAudio, speakingMessageId]);

    useEffect(() => {
        if (!isTtsEnabled) {
            if (audioSourceRef.current) {
                audioSourceRef.current.stop();
                setSpeakingMessageId(null);
            }
            return;
        }

        const lastMessage = messages[messages.length - 1];
        if (lastMessage?.sender === Sender.Bot && lastMessage.text && !lastMessage.isLoading && lastMessage.id !== speakingMessageId) {
            handleSpeakClick(lastMessage.text, lastMessage.id);
        }
    }, [messages, isTtsEnabled, handleSpeakClick, speakingMessageId]);


    useEffect(() => {
        setMessages([{
            id: 'init',
            sender: Sender.Bot,
            text: "Hello! I am PAK AI. I have two modes: 'Chat Assistant' for conversational tasks and 'Consistency Studio' for advanced image control. Use the buttons in the header to switch modes!",
        }]);
    }, []);

    useEffect(() => {
        if (appMode === 'chat') {
            messageListRef.current?.scrollTo({ top: messageListRef.current.scrollHeight, behavior: 'smooth' });
        }
    }, [messages, appMode]);

    useEffect(() => {
        const textarea = textareaRef.current;
        if (textarea) {
            textarea.style.height = 'auto';
            textarea.style.height = `${textarea.scrollHeight}px`;
        }
    }, [input]);

    const handleMicClick = async () => {
        const recognition = recognitionRef.current;
        if (!recognition) return;
    
        if (isRecording) {
            recognition.stop();
        } else {
            try {
                if (navigator.permissions) {
                    const permissionStatus = await navigator.permissions.query({ name: 'microphone' as PermissionName });
                    if (permissionStatus.state === 'denied') {
                        alert("Microphone access is blocked. Please enable it in your browser's site settings to use this feature.");
                        return;
                    }
                }
            } catch (e) {
                console.warn("Could not query permissions API, proceeding with default behavior.", e);
            }
            setInput('');
            recognition.start();
        }
    };

    const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>, isFaceReference: boolean = false) => {
        const files = event.target.files;
        if (files) {
            try {
                const newImages = await Promise.all(Array.from(files).map(async (file: File) => {
                    const base64Data = await fileToBase64(file);
                    return { file, data: base64Data, mimeType: file.type };
                }));
                
                if (isFaceReference) {
                    setFaceReferences(prev => [...prev, ...newImages]);
                    showToast("Face reference added!");
                } else {
                    setUploadedImages(prev => [...prev, ...newImages]);
                }
            } catch (error) {
                console.error("Error processing files:", error);
            }
        }
        if (event.target) event.target.value = ''; // Reset file input
    };
    
    const removeUploadedImage = (indexToRemove: number) => {
        setUploadedImages(prev => prev.filter((_, index) => index !== indexToRemove));
    };

    const removeFaceReference = (indexToRemove: number) => {
        setFaceReferences(prev => prev.filter((_, index) => index !== indexToRemove));
    };

    const handleClearInput = () => {
        setInput('');
        textareaRef.current?.focus();
    };

    const handleSendMessage = useCallback(async (textOverride?: string) => {
        const currentInput = textOverride ?? input;
        if (isProcessing || (!currentInput.trim() && uploadedImages.length === 0)) return;
    
        if (isRecording) recognitionRef.current?.stop();
        setIsProcessing(true);
    
        const userMessageId = Date.now().toString();
        const userMessage: ChatMessage = {
            id: userMessageId,
            sender: Sender.User,
            text: currentInput,
            images: uploadedImages.map(img => `data:${img.mimeType};base64,${img.data}`),
        };
        const botMessageId = (Date.now() + 1).toString();
        const botLoadingMessage: ChatMessage = { id: botMessageId, sender: Sender.Bot, text: '', isLoading: true };

        setMessages(prev => [...prev, userMessage, botLoadingMessage]);
    
        const textToSend = currentInput;
        const imagesToSend = [...faceReferences, ...uploadedImages];
        
        const historyForAPI: Content[] = messages
            .filter(msg => msg.id !== 'init' && !msg.isError && !msg.isLoading)
            .flatMap(msg => {
                const content: Content = {
                    role: msg.sender === Sender.User ? 'user' : 'model',
                    parts: [],
                };

                if (msg.text) {
                    content.parts.push({ text: msg.text });
                }
                
                if (msg.images) {
                    msg.images.forEach(imgDataUrl => {
                        const [header, data] = imgDataUrl.split(',');
                        if (!data) return; 
                        const mimeType = header.match(/:(.*?);/)?.[1] || 'image/png';
                        content.parts.push({ inlineData: { mimeType, data } });
                    });
                }

                return content.parts.length > 0 ? [content] : [];
            });

        setInput('');
        setUploadedImages([]);
    
        try {
            const botResponse = await sendMessage(textToSend, imagesToSend, historyForAPI);

            if (botResponse.resetMemory) {
                setFaceReferences([]);
            }

            if (botResponse.needsAspectRatio && botResponse.pendingPrompt) {
                const finalBotMessage: ChatMessage = {
                    id: botMessageId,
                    sender: Sender.Bot,
                    text: botResponse.text,
                    isLoading: false,
                    needsAspectRatio: true,
                    prompt: botResponse.pendingPrompt,
                };
                setMessages(prev => prev.map(msg => msg.id === botMessageId ? finalBotMessage : msg));
                setIsProcessing(false);
                return;
            }
            
            const finalBotMessage: ChatMessage = {
                id: botMessageId,
                sender: Sender.Bot,
                text: botResponse.text,
                sources: botResponse.sources,
                prompt: botResponse.prompt,
                images: botResponse.image ? [botResponse.image] : undefined,
                isLoading: false,
            };

            setMessages(prev => prev.map(msg => msg.id === botMessageId ? finalBotMessage : msg));
    
        } catch (error) {
            console.error("Error sending message:", error);
            setMessages(prev => prev.map(msg => msg.id === botMessageId ? { ...msg, text: `Error: ${(error as Error).message}`, isLoading: false, isError: true } : msg));
        } finally {
            if (!resizeTask) { 
                 setIsProcessing(false);
            }
        }
    }, [input, isProcessing, uploadedImages, faceReferences, messages, isRecording, resizeTask]);
    
    const handleAspectRatioSelected = useCallback(async (messageId: string, prompt: string, aspectRatio: string) => {
        setIsProcessing(true);
        setMessages(prevMessages => prevMessages.map(msg => 
            msg.id === messageId 
                ? { ...msg, isLoading: true, needsAspectRatio: false, text: `Generating image with aspect ratio ${aspectRatio}...` } 
                : msg
        ));
        try {
            const botResponse = await executePendingImageGeneration(prompt, aspectRatio);
            const finalBotMessage: ChatMessage = {
                id: messageId,
                sender: Sender.Bot,
                text: botResponse.text,
                prompt: botResponse.prompt,
                images: botResponse.image ? [botResponse.image] : undefined,
                isLoading: false,
            };
            setMessages(prevMessages => prevMessages.map(msg => msg.id === messageId ? finalBotMessage : msg));
        } catch (error) {
            console.error("Error executing pending generation:", error);
            setMessages(prevMessages => prevMessages.map(msg => 
                msg.id === messageId 
                ? { ...msg, text: `Error: ${(error as Error).message}`, isLoading: false, isError: true, needsAspectRatio: false } 
                : msg
            ));
        } finally {
            setIsProcessing(false);
        }
    }, []);

    const handleResizeClick = (message: ChatMessage) => setResizeTask(message);

    const handleReuseImage = (message: ChatMessage) => {
        if (message.images && message.images.length > 0) {
            const imageToReuse = message.images[0];
            const [header, data] = imageToReuse.split(',');
            const mimeType = header.match(/:(.*?);/)?.[1] || 'image/png';
            const byteCharacters = atob(data);
            const byteNumbers = new Array(byteCharacters.length);
            for (let i = 0; i < byteCharacters.length; i++) {
                byteNumbers[i] = byteCharacters.charCodeAt(i);
            }
            const byteArray = new Uint8Array(byteNumbers);
            const blob = new Blob([byteArray], { type: mimeType });
            const file = new File([blob], "reused-image.png", { type: mimeType });

            setUploadedImages(prev => [...prev, { file, data, mimeType }]);
            if (appMode === 'chat') textareaRef.current?.focus();
        }
    };
    
    const showToast = (text: string) => {
        setToastText(text);
        setTimeout(() => setToastText(null), 2000);
    };

    const handleCopyPrompt = (prompt: string) => {
        if (prompt) {
            navigator.clipboard.writeText(prompt);
            showToast("Prompt copied to clipboard!");
        }
    };

    const handleCopyResponse = (text: string) => {
        if (text) {
            navigator.clipboard.writeText(text);
            showToast("Response copied to clipboard!");
        }
    };
    
    const handleDownloadClick = (message: ChatMessage) => {
        if (message.images && message.images.length > 0) {
            const imageSrc = message.images[0];
            const link = document.createElement('a');
            link.href = imageSrc;
            link.download = `pak-ai-image-${Date.now()}.png`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    };

    const handleExecuteResize = useCallback(async (aspectRatio: string) => {
        if (!resizeTask || !resizeTask.images || resizeTask.images.length === 0) return;
        setIsProcessing(true);
        
        const { prompt: originalPrompt } = resizeTask;
        const originalImageSrc = resizeTask.images[0];
        setResizeTask(null);

        const botMessageId = (Date.now()).toString();
        const botLoadingMessage: ChatMessage = { id: botMessageId, sender: Sender.Bot, text: '', isLoading: true };
        setMessages(prev => [...prev, botLoadingMessage]);
        
        try {
            let botResponse;
            // CRITICAL CHANGE: Always prefer the image data if available for resizing
            // This ensures we preserve visual details (Face, Dress, Env) via Image-to-Image
            if (originalImageSrc) {
                 const [header, data] = originalImageSrc.split(',');
                 const mimeType = header.match(/:(.*?);/)?.[1] || 'image/png';
                 const imageToResize = { data, mimeType };
                 const res = await resizeImage(imageToResize, aspectRatio);
                 botResponse = { ...res, prompt: originalPrompt || 'resized' };
            } else if (originalPrompt) {
                // Fallback to text-only generation only if image data is somehow missing
                botResponse = await executePendingImageGeneration(originalPrompt, aspectRatio);
            } else {
                throw new Error("No image or prompt found to resize.");
            }
            
             const finalBotMessage: ChatMessage = {
                id: botMessageId,
                sender: Sender.Bot,
                text: botResponse.text,
                prompt: botResponse.prompt,
                images: botResponse.image ? [botResponse.image] : undefined,
                isLoading: false,
            };
            setMessages(prev => prev.map(msg => msg.id === botMessageId ? finalBotMessage : msg));
        } catch (error) {
            console.error("Error resizing image:", error);
            setMessages(prev => prev.map(msg => msg.id === botMessageId ? { ...msg, text: `Error: ${(error as Error).message}`, isLoading: false, isError: true } : msg));
        } finally {
            setIsProcessing(false);
        }
    }, [resizeTask]);
    
    const handleLikeDislike = useCallback((messageId: string, feedback: 'liked' | 'disliked') => {
        setMessages(prev => prev.map(msg => {
            if (msg.id === messageId) {
                return { ...msg, feedback: msg.feedback === feedback ? undefined : feedback };
            }
            return msg;
        }));
        showToast("Feedback submitted!");
    }, []);
    
    const handleShareClick = useCallback(async (text: string) => {
        if (navigator.share) {
            try { await navigator.share({ title: 'PAK AI Response', text: text }); } catch (error) { console.error('Error sharing:', error); }
        } else { handleCopyResponse(text); }
    }, []);

    const handleRegenerate = useCallback(async (messageId: string) => {
        const botMessageIndex = messages.findIndex(msg => msg.id === messageId);
        if (botMessageIndex < 1) return;
        let userMessageIndex = -1;
        for (let i = botMessageIndex - 1; i >= 0; i--) {
            if (messages[i].sender === Sender.User) { userMessageIndex = i; break; }
        }
        if (userMessageIndex === -1) return;
        const userMessage = messages[userMessageIndex];
        const history = messages.slice(0, userMessageIndex);
        setIsProcessing(true);
        setMessages(prev => prev.map(msg => msg.id === messageId ? { ...msg, isLoading: true, text: '', images: undefined, sources: undefined, prompt: undefined, isError: false, feedback: undefined } : msg));
        const textToSend = userMessage.text;
        const imagesToSend = [...faceReferences, ...(userMessage.images || []).map(imgDataUrl => {
            const [header, data] = imgDataUrl.split(',');
            const mimeType = header.match(/:(.*?);/)?.[1] || 'image/png';
            return { data, mimeType };
        })];
        const historyForAPI: Content[] = history
            .filter(msg => msg.id !== 'init' && !msg.isError && !msg.isLoading)
            .flatMap(msg => {
                const content: Content = { role: msg.sender === Sender.User ? 'user' : 'model', parts: [] };
                if (msg.text) content.parts.push({ text: msg.text });
                if (msg.images) {
                    msg.images.forEach(imgDataUrl => {
                        const [header, data] = imgDataUrl.split(',');
                        if (data) {
                            const mimeType = header.match(/:(.*?);/)?.[1] || 'image/png';
                            content.parts.push({ inlineData: { mimeType, data } });
                        }
                    });
                }
                return content.parts.length > 0 ? [content] : [];
            });
        try {
            const botResponse = await sendMessage(textToSend, imagesToSend, historyForAPI);
            if (botResponse.resetMemory) setFaceReferences([]);
            if (botResponse.needsAspectRatio && botResponse.pendingPrompt) {
                const updatedMessage: ChatMessage = { id: messageId, sender: Sender.Bot, text: botResponse.text, isLoading: false, needsAspectRatio: true, prompt: botResponse.pendingPrompt };
                setMessages(prev => prev.map(msg => msg.id === messageId ? updatedMessage : msg));
                setIsProcessing(false);
                return;
            }
            const finalBotMessage: ChatMessage = { id: messageId, sender: Sender.Bot, text: botResponse.text, sources: botResponse.sources, prompt: botResponse.prompt, images: botResponse.image ? [botResponse.image] : undefined, isLoading: false };
            setMessages(prev => prev.map(msg => msg.id === messageId ? finalBotMessage : msg));
        } catch (error) {
            console.error("Error regenerating response:", error);
            setMessages(prev => prev.map(msg => msg.id === messageId ? { ...msg, text: `Error: ${(error as Error).message}`, isLoading: false, isError: true } : msg));
        } finally { setIsProcessing(false); }
    }, [messages, faceReferences]);

    return (
        <div className="flex flex-col h-screen bg-gray-900 text-gray-100 font-sans">
            {toastText && <Toast message={toastText} />}
            
            {resizeTask && (
                 <Modal title="Resize Image" description="Select a new aspect ratio." onCancel={() => setResizeTask(null)}>
                    <div className="grid grid-cols-2 gap-4">
                        {ASPECT_RATIOS.map(({ ratio, label, description }) => (
                            <button key={ratio} onClick={() => handleExecuteResize(ratio)} className="flex flex-col items-center text-left p-4 bg-gray-700 border border-gray-600 rounded-xl hover:border-blue-500 transition-all">
                                <div className="w-full mb-2 text-center text-xs text-gray-400">{ratio}</div>
                                <span className="block font-bold text-white text-sm">{label}</span>
                                <span className="block text-xs text-gray-400">{description}</span>
                            </button>
                        ))}
                    </div>
                </Modal>
            )}

            <header className="bg-gray-900 border-b border-gray-800 shadow-lg p-4 flex items-center justify-between z-10">
                <div className="flex gap-2">
                     <button 
                        onClick={() => setAppMode('chat')}
                        className={`flex items-center gap-2 px-3 py-1.5 rounded-lg transition-colors ${appMode === 'chat' ? 'bg-blue-600 text-white' : 'text-gray-400 hover:bg-gray-800'}`}
                    >
                        <ChatBubbleIcon /> Chat
                    </button>
                    <button 
                        onClick={() => setAppMode('studio')}
                        className={`flex items-center gap-2 px-3 py-1.5 rounded-lg transition-colors ${appMode === 'studio' ? 'bg-blue-600 text-white' : 'text-gray-400 hover:bg-gray-800'}`}
                    >
                        <DashboardIcon /> Studio
                    </button>
                </div>

                <h1 className="text-xl font-bold text-white flex items-center gap-2 absolute left-1/2 -translate-x-1/2 pointer-events-none">
                    <SparklesIcon /> PAK AI
                </h1>

                <div className="flex justify-end">
                    <button onClick={() => setIsTtsEnabled(prev => !prev)} className={`p-2 rounded-full transition-colors ${isTtsEnabled ? 'text-blue-400 bg-gray-800' : 'text-gray-400 hover:bg-gray-800'}`}>
                        {isTtsEnabled ? <SpeakerWaveIcon /> : <SpeakerXMarkIcon />}
                    </button>
                </div>
            </header>
    
            {appMode === 'studio' ? (
                <div className="flex-1 overflow-y-auto">
                    <ConsistencyStudio consistencyAI={consistencyAI} />
                </div>
            ) : (
                <>
                    <main ref={messageListRef} className="flex-1 overflow-y-auto p-6 space-y-4">
                        {messages.map((msg, index) => {
                            let promptToCopy: string | undefined = undefined;
                            if (msg.sender === Sender.Bot) {
                                if (msg.prompt && msg.prompt !== 'resized') {
                                    promptToCopy = msg.prompt;
                                } else {
                                    for (let i = index - 1; i >= 0; i--) {
                                        if (messages[i].sender === Sender.User) { promptToCopy = messages[i].text; break; }
                                    }
                                }
                            }
                            return (
                                <Message
                                    key={msg.id} message={msg} promptToCopy={promptToCopy} isSpeaking={speakingMessageId === msg.id}
                                    onResizeClick={handleResizeClick} onReuseClick={handleReuseImage} onCopyPromptClick={handleCopyPrompt}
                                    onCopyResponseClick={handleCopyResponse} onDownloadClick={handleDownloadClick} onAspectRatioSelect={handleAspectRatioSelected}
                                    onLikeClick={(id) => handleLikeDislike(id, 'liked')} onDislikeClick={(id) => handleLikeDislike(id, 'disliked')}
                                    onSpeakClick={handleSpeakClick} onRegenerateClick={handleRegenerate} onShareClick={handleShareClick}
                                />
                            );
                        })}
                    </main>
            
                    <footer className="bg-gray-900 p-4 border-t border-gray-800">
                        <div className="max-w-4xl mx-auto">
                            {faceReferences.length > 0 && (
                                <div className="p-2 mb-2 border-b border-gray-800">
                                    <div className="text-xs text-gray-400 mb-2 uppercase font-semibold tracking-wider">Face Memory (Chat)</div>
                                    <div className="flex gap-2 overflow-x-auto pb-1">
                                        {faceReferences.map((image, index) => (
                                            <div key={index} className="relative w-14 h-14 flex-shrink-0 group">
                                                <img src={`data:${image.mimeType};base64,${image.data}`} alt={`face ref ${index + 1}`} className="w-full h-full object-cover rounded-full border-2 border-blue-500"/>
                                                <button onClick={() => removeFaceReference(index)} className="absolute -top-1 -right-1 bg-gray-800 rounded-full text-red-400 hover:text-red-300 transition-colors shadow-sm"><XCircleIcon /></button>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                            
                            {uploadedImages.length > 0 && (
                                <div className="p-2 mb-2 border-b border-gray-800">
                                    <div className="flex gap-3 overflow-x-auto">
                                        {uploadedImages.map((image, index) => (
                                            <div key={index} className="relative w-20 h-20 flex-shrink-0 group">
                                                <img src={`data:${image.mimeType};base64,${image.data}`} alt={`upload preview ${index + 1}`} className="w-full h-full object-cover rounded-lg"/>
                                                <button onClick={() => removeUploadedImage(index)} className="absolute -top-2 -right-2 bg-gray-800 rounded-full text-gray-400 hover:text-white transition-colors"><XCircleIcon /></button>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                            <div className="relative bg-gray-800 rounded-2xl p-2 flex items-end border border-gray-700">
                                <div className="flex gap-1 mr-1">
                                    <button onClick={() => fileInputRef.current?.click()} className="p-2 text-gray-400 hover:text-white transition-colors" aria-label="Attach file" disabled={isProcessing}><PaperClipIcon /></button>
                                    <button onClick={() => faceInputRef.current?.click()} className="p-2 text-gray-400 hover:text-blue-400 transition-colors relative group" aria-label="Add Face Reference" disabled={isProcessing}>
                                        <AddUserIcon />
                                        <span className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 bg-black text-white text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none">Add Face Ref</span>
                                    </button>
                                </div>
                                <input type="file" ref={fileInputRef} onChange={(e) => handleFileChange(e, false)} className="hidden" accept="image/*" multiple />
                                <input type="file" ref={faceInputRef} onChange={(e) => handleFileChange(e, true)} className="hidden" accept="image/*" multiple />
            
                                <textarea
                                    ref={textareaRef} value={input} onChange={(e) => setInput(e.target.value)}
                                    onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSendMessage(); } }}
                                    placeholder={isRecording ? "Listening..." : "Chat..."}
                                    rows={1} maxLength={MAX_CHAR_LIMIT}
                                    className="flex-1 bg-transparent text-gray-100 placeholder-gray-500 focus:outline-none resize-none px-2 max-h-64"
                                    disabled={isProcessing}
                                />

                                {input.trim() && !isProcessing && (
                                    <button onClick={handleClearInput} className="p-2 text-gray-400 hover:text-white transition-colors"><XCircleIcon /></button>
                                )}

                                <button onClick={handleMicClick} disabled={isProcessing || !recognitionRef.current} className={`p-2 rounded-full text-white transition-colors ${isRecording ? 'bg-red-600' : 'bg-gray-700 hover:bg-gray-600'} disabled:opacity-50`}>
                                <MicrophoneIcon />
                                </button>

                                <button onClick={() => handleSendMessage()} disabled={isProcessing || (!input.trim() && uploadedImages.length === 0)} className="ml-2 p-2 rounded-full bg-blue-600 text-white disabled:bg-gray-600 disabled:cursor-not-allowed hover:bg-blue-700 transition-colors">
                                    {isProcessing ? <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div> : <PaperAirplaneIcon />}
                                </button>
                            </div>
                        </div>
                    </footer>
                </>
            )}
        </div>
    );
}
