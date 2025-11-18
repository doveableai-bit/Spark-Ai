import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { Sender } from './types';
import type { ChatMessage } from './types';
import { sendMessage, executePendingImageGeneration, resizeImage } from './services/geminiService';
import { Content } from '@google/genai';


declare global {
    interface Window {
        marked: {
            parse(markdown: string): string;
        };
        SpeechRecognition: any;
        webkitSpeechRecognition: any;
    }
}

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


// --- SVG Icons ---
const SparklesIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
        <path fillRule="evenodd" d="M10.868 2.884c.321-.772 1.415-.772 1.736 0l1.291 3.118c.261.63.852 1.054 1.523 1.12l3.434.5c1.13.164 1.588 1.555.744 2.368l-2.484 2.42c-.469.458-.682 1.12-.549 1.758l.586 3.42a1.363 1.363 0 0 1-1.98 1.432L10 16.45l-3.056 1.606a1.363 1.363 0 0 1-1.98-1.432l.586-3.42a1.99 1.99 0 0 0-.55-1.758l-2.484-2.42c-.844-.813-.386-2.204.744-2.368l3.434-.5c.671-.066 1.262-.49 1.523-1.12l1.29-3.118Z" clipRule="evenodd" />
    </svg>
);

const UserIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
        <path fillRule="evenodd" d="M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0Zm-5.5-2.5a2.5 2.5 0 1 1-5 0 2.5 2.5 0 0 1 5 0ZM10 12a5.99 5.99 0 0 0-4.793 2.39A6.483 6.483 0 0 0 10 16.5a6.483 6.483 0 0 0 4.793-2.11A5.99 5.99 0 0 0 10 12Z" clipRule="evenodd" />
    </svg>
);

const PaperAirplaneIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
        <path d="M3.105 2.289a.75.75 0 0 0-.826.95l1.414 4.949a.75.75 0 0 0 .95.54l3.118-1.559a.75.75 0 0 1 .527.955l-2.093 4.185a.75.75 0 0 0 .504.986l5.05 1.443a.75.75 0 0 0 .986-.504l4.185-10.462a.75.75 0 0 0-.527-.955L4.413 3.457a.75.75 0 0 0-.95.54l-1.414-1.718Z" />
        <path d="M12.91 8.351a.75.75 0 0 0-1.06 0l-2.093 2.093a.75.75 0 0 0 1.06 1.06L12.91 9.41a.75.75 0 0 0 0-1.06Z" />
    </svg>
);

const PaperClipIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
        <path fillRule="evenodd" d="M15.621 4.379a3 3 0 0 0-4.242 0l-7 7a3 3 0 0 0 4.241 4.243h.001l.497-.5a.75.75 0 0 1 1.06 1.06l-.497.5a4.5 4.5 0 1 1-6.364-6.364l7-7a4.5 4.5 0 0 1 6.364 6.364l-3.182 3.182a.75.75 0 0 0 1.061 1.06l3.182-3.182a3 3 0 0 0 0-4.242Z" clipRule="evenodd" />
    </svg>
);

const XCircleIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
        <path fillRule="evenodd" d="M10 18a8 8 0 1 0 0-16 8 8 0 0 0 0 16Zm3.707-10.707a1 1 0 0 0-1.414-1.414L10 8.586 7.707 6.293a1 1 0 0 0-1.414 1.414L8.586 10l-2.293 2.293a1 1 0 1 0 1.414 1.414L10 11.414l2.293 2.293a1 1 0 0 0 1.414-1.414L11.414 10l2.293-2.293Z" clipRule="evenodd" />
    </svg>
);

const MicrophoneIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
      <path d="M7 4a3 3 0 0 1 6 0v6a3 3 0 1 1-6 0V4Z" />
      <path d="M5.5 8.5a.5.5 0 0 1 .5.5v1.5a.5.5 0 0 1-1 0V9a.5.5 0 0 1 .5-.5Z" />
      <path d="M10 18a.5.5 0 0 0 .5-.5v-2.29a6.474 6.474 0 0 0 5-6.21v-1.5a.5.5 0 0 0-1 0v1.5a5.474 5.474 0 0 1-10 0v-1.5a.5.5 0 0 0-1 0v1.5a6.474 6.474 0 0 0 5 6.21v2.29a.5.5 0 0 0 .5.5Z" />
    </svg>
);

const SpeakerWaveIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5">
      <path d="M13.5 4.06c0-1.336-1.616-2.005-2.56-1.06l-4.5 4.5H4.508c-1.141 0-2.318.664-2.66 1.905A9.76 9.76 0 0 0 1.5 12c0 .898.121 1.768.348 2.595.341 1.24 1.518 1.905 2.66 1.905H6.44l4.5 4.5c.945.945 2.56.276 2.56-1.06V4.06zM18.584 5.106a.75.75 0 0 1 1.06 0c3.807 3.808 3.807 9.98 0 13.788a.75.75 0 0 1-1.06-1.06 8.25 8.25 0 0 0 0-11.668.75.75 0 0 1 0-1.06z" />
      <path d="M15.932 7.757a.75.75 0 0 1 1.061 0 6 6 0 0 1 0 8.486.75.75 0 0 1-1.06-1.061 4.5 4.5 0 0 0 0-6.364.75.75 0 0 1 0-1.06z" />
    </svg>
);

const SpeakerXMarkIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5">
      <path d="M13.5 4.06c0-1.336-1.616-2.005-2.56-1.06l-4.5 4.5H4.508c-1.141 0-2.318.664-2.66 1.905A9.76 9.76 0 0 0 1.5 12c0 .898.121 1.768.348 2.595.341 1.24 1.518 1.905 2.66 1.905H6.44l4.5 4.5c.945.945 2.56.276 2.56-1.06V4.06zM17.78 9.22a.75.75 0 1 0-1.06 1.06L18.94 12l-2.22 2.22a.75.75 0 1 0 1.06 1.06L20 13.06l2.22 2.22a.75.75 0 1 0 1.06-1.06L21.06 12l2.22-2.22a.75.75 0 1 0-1.06-1.06L20 10.94l-2.22-2.22z" />
    </svg>
);
const ResizeIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
      <path d="M10 3a.75.75 0 0 1 .75.75v1.5h1.5a.75.75 0 0 1 0 1.5h-1.5v1.5a.75.75 0 0 1-1.5 0v-1.5h-1.5a.75.75 0 0 1 0-1.5h1.5v-1.5A.75.75 0 0 1 10 3ZM10 17a.75.75 0 0 1-.75-.75v-1.5h-1.5a.75.75 0 0 1 0-1.5h1.5v-1.5a.75.75 0 0 1 1.5 0v1.5h1.5a.75.75 0 0 1 0 1.5h-1.5v1.5A.75.75 0 0 1 10 17ZM17 10a.75.75 0 0 1-.75.75h-1.5v1.5a.75.75 0 0 1-1.5 0v-1.5h-1.5a.75.75 0 0 1 0-1.5h1.5v-1.5a.75.75 0 0 1 1.5 0v1.5h1.5A.75.75 0 0 1 17 10ZM3 10a.75.75 0 0 1 .75-.75h1.5v-1.5a.75.75 0 0 1 1.5 0v1.5h1.5a.75.75 0 0 1 0 1.5h-1.5v1.5a.75.75 0 0 1-1.5 0v-1.5h-1.5A.75.75 0 0 1 3 10Z" />
    </svg>
);
const ReuseIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
      <path fillRule="evenodd" d="M15.312 11.424a5.5 5.5 0 0 1-9.201-4.42 5.5 5.5 0 0 1 10.852 2.132.75.75 0 0 1-1.423.433A4 4 0 0 0 6.13 8.355a4 4 0 0 0 7.938 1.48.75.75 0 0 1 1.244.789ZM4.688 8.576a.75.75 0 0 1 1.423-.433 4 4 0 0 0 8.239-1.523.75.75 0 0 1 1.244-.789 5.5 5.5 0 0 1-10.852-2.132.75.75 0 0 1 1.423-.433A4 4 0 0 0 13.87 11.645a4 4 0 0 0-7.938-1.48.75.75 0 0 1-1.244-.789Z" clipRule="evenodd" />
    </svg>
);
const CopyIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
      <path d="M7 3.5A1.5 1.5 0 0 1 8.5 2h3.879a1.5 1.5 0 0 1 1.06.44l3.122 3.121A1.5 1.5 0 0 1 17 6.621V16.5a1.5 1.5 0 0 1-1.5 1.5h-7A1.5 1.5 0 0 1 7 16.5v-13Z" />
      <path d="M4.5 6A1.5 1.5 0 0 0 3 7.5v10A1.5 1.5 0 0 0 4.5 19h7a1.5 1.5 0 0 0 1.5-1.5v-2a.75.75 0 0 0-1.5 0v2A.5.5 0 0 1 11.5 18h-7a.5.5 0 0 1-.5-.5v-10a.5.5 0 0 1 .5-.5h2a.75.75 0 0 0 0-1.5h-2Z" />
    </svg>
);

const DownloadIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
      <path d="M10.75 2.75a.75.75 0 0 0-1.5 0v8.614L6.295 8.235a.75.75 0 1 0-1.09 1.03l4.25 4.5a.75.75 0 0 0 1.09 0l4.25-4.5a.75.75 0 0 0-1.09-1.03l-2.955 3.129V2.75Z" />
      <path d="M3.5 12.75a.75.75 0 0 0-1.5 0v2.5A2.75 2.75 0 0 0 4.75 18h10.5A2.75 2.75 0 0 0 18 15.25v-2.5a.75.75 0 0 0-1.5 0v2.5c0 .69-.56 1.25-1.25 1.25H4.75c-.69 0-1.25-.56-1.25-1.25v-2.5Z" />
    </svg>
);


// --- Child Components ---

interface MessageProps {
    message: ChatMessage;
    onResizeClick: (message: ChatMessage) => void;
    onReuseClick: (message: ChatMessage) => void;
    onCopyPromptClick: (message: ChatMessage) => void;
    onDownloadClick: (message: ChatMessage) => void;
    onAspectRatioSelect: (messageId: string, prompt: string, ratio: string) => void;
}

const ActionButton: React.FC<{ onClick: () => void; icon: React.ReactNode; label: string }> = ({ onClick, icon, label }) => (
    <button
        onClick={onClick}
        className="flex items-center gap-1.5 text-xs text-gray-300 bg-gray-600/50 hover:bg-gray-600 px-2 py-1 rounded-md transition-colors"
        aria-label={label}
    >
        {icon}
        {label}
    </button>
);

const Message: React.FC<MessageProps> = ({ message, onResizeClick, onReuseClick, onCopyPromptClick, onDownloadClick, onAspectRatioSelect }) => {
    const isUser = message.sender === Sender.User;
    const Icon = isUser ? UserIcon : SparklesIcon;

    const parsedText = useMemo(() => {
        if (!message.text || typeof window.marked === 'undefined') {
            return { __html: message.text.replace(/\n/g, '<br />') };
        }
        return { __html: window.marked.parse(message.text) };
    }, [message.text]);

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

                        {!isUser && message.images && message.images.length > 0 && message.prompt && (
                            <div className="flex items-center flex-wrap gap-2 mt-3 pt-3 border-t border-gray-600">
                                <ActionButton onClick={() => onResizeClick(message)} icon={<ResizeIcon />} label="Resize" />
                                <ActionButton onClick={() => onReuseClick(message)} icon={<ReuseIcon />} label="Reuse" />
                                <ActionButton onClick={() => onCopyPromptClick(message)} icon={<CopyIcon />} label="Copy Prompt" />
                                <ActionButton onClick={() => onDownloadClick(message)} icon={<DownloadIcon />} label="Download" />
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

                        {message.sources && message.sources.length > 0 && (
                            <div className="mt-4 pt-2 border-t border-gray-600">
                                <h4 className="text-xs font-semibold text-gray-400 mb-1">Sources:</h4>
                                <div className="flex flex-wrap gap-2">
                                    {message.sources.map((source, index) => (
                                        <a key={index} href={source.web.uri} target="_blank" rel="noopener noreferrer"
                                           className="text-xs bg-gray-600 hover:bg-gray-500 text-blue-300 px-2 py-1 rounded-md transition-colors">
                                            {source.web.title || new URL(source.web.uri).hostname}
                                        </a>
                                    ))}
                                </div>
                            </div>
                        )}
                        {message.isError && <p className="text-red-400 text-sm mt-2">An error occurred.</p>}
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


interface ModalProps {
    title: string;
    description?: string;
    onCancel: () => void;
    children: React.ReactNode;
}

const Modal: React.FC<ModalProps> = ({ title, description, onCancel, children }) => (
    <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 transition-opacity">
        <div className="bg-gray-800 rounded-2xl shadow-xl p-6 w-full max-w-md mx-4">
            <h2 className="text-xl font-bold text-white mb-2">{title}</h2>
            {description && <p className="text-gray-400 mb-6">{description}</p>}
            <div className="space-y-3">
                {children}
            </div>
            <button
                onClick={onCancel}
                className="w-full mt-6 p-2 bg-gray-600 hover:bg-gray-500 rounded-lg text-white font-semibold transition-colors"
            >
                Cancel
            </button>
        </div>
    </div>
);

const Toast: React.FC<{ message: string }> = ({ message }) => (
    <div className="fixed bottom-24 left-1/2 -translate-x-1/2 bg-green-600 text-white px-4 py-2 rounded-lg shadow-lg animate-toast z-50">
        {message}
    </div>
);


// --- Main App Component ---

type UploadedImage = { file: File, data: string, mimeType: string };

export default function App() {
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [input, setInput] = useState('');
    const [isProcessing, setIsProcessing] = useState(false);
    const [isRecording, setIsRecording] = useState(false);
    const [uploadedImages, setUploadedImages] = useState<UploadedImage[]>([]);
    const [isTtsEnabled, setIsTtsEnabled] = useState(false);
    const [resizeTask, setResizeTask] = useState<ChatMessage | null>(null);
    const [toastText, setToastText] = useState<string | null>(null);

    const recognitionRef = useRef<any>(null);
    const messageListRef = useRef<HTMLDivElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    useEffect(() => {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
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

    useEffect(() => {
        if (!isTtsEnabled) {
            window.speechSynthesis.cancel();
            return;
        }

        const lastMessage = messages[messages.length - 1];
        if (lastMessage?.sender === Sender.Bot && lastMessage.text && !lastMessage.isLoading) {
            const cleanTextForSpeech = (text: string) => {
                return text
                    .replace(/```[\s\S]*?```/g, '(Code block is displayed on screen.)')
                    .replace(/\[([^\]]+)\]\([^\)]+\)/g, '$1')
                    .replace(/(\*|_|`)/g, '');
            };
            const textToSpeak = cleanTextForSpeech(lastMessage.text);
            const utterance = new SpeechSynthesisUtterance(textToSpeak);
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(utterance);
        }
    }, [messages, isTtsEnabled]);


    useEffect(() => {
        setMessages([{
            id: 'init',
            sender: Sender.Bot,
            text: "Hello! I am PAK AI, your multi-modal assistant. You can chat with me, ask me to generate or edit images, or search the web. How can I help you today?",
        }]);
    }, []);

    useEffect(() => {
        messageListRef.current?.scrollTo({ top: messageListRef.current.scrollHeight, behavior: 'smooth' });
    }, [messages]);

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

    const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
        const files = event.target.files;
        if (files) {
            try {
                const newImages = await Promise.all(Array.from(files).map(async (file: File) => {
                    const base64Data = await fileToBase64(file);
                    return { file, data: base64Data, mimeType: file.type };
                }));
                setUploadedImages(prev => [...prev, ...newImages]);
            } catch (error) {
                console.error("Error processing files:", error);
            }
        }
        if (event.target) event.target.value = ''; // Reset file input
    };
    
    const removeUploadedImage = (indexToRemove: number) => {
        setUploadedImages(prev => prev.filter((_, index) => index !== indexToRemove));
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
        const imagesToSend = uploadedImages;
        
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
                        if (!data) return; // Skip if data url is malformed
                        const mimeType = header.match(/:(.*?);/)?.[1] || 'image/png';
                        content.parts.push({ inlineData: { mimeType, data } });
                    });
                }

                // The API expects parts to not be empty
                return content.parts.length > 0 ? [content] : [];
            });

        setInput('');
        setUploadedImages([]);
    
        try {
            const botResponse = await sendMessage(textToSend, imagesToSend, historyForAPI);

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
            if (!resizeTask) { // Don't reset processing if a resize modal is active
                 setIsProcessing(false);
            }
        }
    }, [input, isProcessing, uploadedImages, messages, isRecording, resizeTask]);
    
    const handleAspectRatioSelected = useCallback(async (messageId: string, prompt: string, aspectRatio: string) => {
        setIsProcessing(true);
    
        // Update the UI to show a loading state for the specific message, using a functional update
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
            // Final update, also functional
            setMessages(prevMessages => prevMessages.map(msg => msg.id === messageId ? finalBotMessage : msg));
        } catch (error) {
            console.error("Error executing pending generation:", error);
            // Error update, also functional
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
            textareaRef.current?.focus();
        }
    };
    const handleCopyPromptClick = (message: ChatMessage) => {
        if (message.prompt) {
            navigator.clipboard.writeText(message.prompt);
            setToastText("Prompt copied to clipboard!");
            setTimeout(() => setToastText(null), 2000);
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
        
        const originalImageSrc = resizeTask.images[0];
        const { prompt: originalPrompt } = resizeTask;
        setResizeTask(null);

        const [header, data] = originalImageSrc.split(',');
        const mimeType = header.match(/:(.*?);/)?.[1] || 'image/png';
        const imageToResize = { data, mimeType };

        const botMessageId = (Date.now()).toString();
        const botLoadingMessage: ChatMessage = { id: botMessageId, sender: Sender.Bot, text: '', isLoading: true };
        setMessages(prev => [...prev, botLoadingMessage]);
        
        try {
            const botResponse = await resizeImage(imageToResize, aspectRatio);
             const finalBotMessage: ChatMessage = {
                id: botMessageId,
                sender: Sender.Bot,
                text: botResponse.text,
                prompt: originalPrompt || undefined,
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

    return (
        <div className="flex flex-col h-screen bg-gray-800 text-gray-100 font-sans">
            {toastText && <Toast message={toastText} />}
            
            {resizeTask && (
                 <Modal
                    title="Select a New Aspect Ratio"
                    onCancel={() => setResizeTask(null)}
                >
                    {ASPECT_RATIOS.map(({ ratio, label, description }) => (
                         <button key={ratio} onClick={() => handleExecuteResize(ratio)} className="w-full text-left p-4 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors">
                            <p className="font-semibold text-white">{label} <span className="text-gray-400 font-normal">({ratio})</span></p>
                            <p className="text-sm text-gray-400">{description}</p>
                        </button>
                    ))}
                </Modal>
            )}
            <header className="bg-gray-900 shadow-lg p-4 grid grid-cols-3 items-center z-10">
                <div />
                <h1 className="text-xl font-bold text-white flex items-center gap-2 justify-center col-start-2">
                    <SparklesIcon /> PAK AI
                </h1>
                <div className="flex justify-end">
                    <button
                        onClick={() => setIsTtsEnabled(prev => !prev)}
                        className={`p-2 rounded-full transition-colors ${isTtsEnabled ? 'text-blue-400 bg-gray-700' : 'text-gray-400 hover:bg-gray-700'}`}
                        aria-label={isTtsEnabled ? "Disable text-to-speech" : "Enable text-to-speech"}
                    >
                        {isTtsEnabled ? <SpeakerWaveIcon /> : <SpeakerXMarkIcon />}
                    </button>
                </div>
            </header>
    
            <main ref={messageListRef} className="flex-1 overflow-y-auto p-6 space-y-4">
                 {messages.map((msg) => (
                    <Message
                        key={msg.id}
                        message={msg}
                        onResizeClick={handleResizeClick}
                        onReuseClick={handleReuseImage}
                        onCopyPromptClick={handleCopyPromptClick}
                        onDownloadClick={handleDownloadClick}
                        onAspectRatioSelect={handleAspectRatioSelected}
                    />
                ))}
            </main>
    
            <footer className="bg-gray-900 p-4 border-t border-gray-700">
                <div className="max-w-4xl mx-auto">
                     {uploadedImages.length > 0 && (
                        <div className="p-2 mb-2 border-b border-gray-700">
                            <div className="flex gap-3 overflow-x-auto">
                                {uploadedImages.map((image, index) => (
                                    <div key={index} className="relative w-20 h-20 flex-shrink-0 group">
                                        <img src={`data:${image.mimeType};base64,${image.data}`} alt={`upload preview ${index + 1}`} className="w-full h-full object-cover rounded-lg"/>
                                        <button 
                                            onClick={() => removeUploadedImage(index)} 
                                            className="absolute -top-2 -right-2 bg-gray-800 rounded-full text-gray-400 hover:text-white transition-colors" 
                                            aria-label="Remove image"
                                        >
                                            <XCircleIcon />
                                        </button>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                    <div className="relative bg-gray-700 rounded-2xl p-2 flex items-end">
                        <button onClick={() => fileInputRef.current?.click()} className="p-2 text-gray-400 hover:text-white transition-colors" aria-label="Attach file" disabled={isProcessing}>
                            <PaperClipIcon />
                        </button>
                        <input type="file" ref={fileInputRef} onChange={handleFileChange} className="hidden" accept="image/*" multiple />
    
                        <textarea
                            ref={textareaRef}
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSendMessage(); } }}
                            placeholder={isRecording ? "Listening..." : "Chat with PAK AI, or upload an image..."}
                            rows={1}
                            maxLength={MAX_CHAR_LIMIT}
                            className="flex-1 bg-transparent text-gray-100 placeholder-gray-400 focus:outline-none resize-none px-2 max-h-64"
                            disabled={isProcessing}
                        />

                        {input.trim() && !isProcessing && (
                            <button onClick={handleClearInput} className="p-2 text-gray-400 hover:text-white transition-colors" aria-label="Clear input">
                                <XCircleIcon />
                            </button>
                        )}

                        <button
                            onClick={handleMicClick}
                            disabled={isProcessing || !recognitionRef.current}
                            className={`p-2 rounded-full text-white transition-colors ${isRecording ? 'bg-red-600' : 'bg-gray-600 hover:bg-gray-500'} disabled:bg-gray-500 disabled:cursor-not-allowed`}
                            aria-label={isRecording ? "Stop recording" : "Start recording"}
                        >
                           <MicrophoneIcon />
                        </button>

                        <button
                            onClick={() => handleSendMessage()}
                            disabled={isProcessing || (!input.trim() && uploadedImages.length === 0)}
                            className="ml-2 p-2 rounded-full bg-blue-600 text-white disabled:bg-gray-500 disabled:cursor-not-allowed hover:bg-blue-700 transition-colors"
                            aria-label="Send message"
                        >
                            {isProcessing ? <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div> : <PaperAirplaneIcon />}
                        </button>
                    </div>
                </div>
            </footer>
        </div>
    );
}