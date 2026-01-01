import { useState, useEffect } from 'react';
import { Play, CheckCircle, Lock } from 'lucide-react';

interface VideoPlayerProps {
  url: string;
  title: string;
  isCompleted: boolean;
  isLocked: boolean;
  onComplete: () => void;
}

export function VideoPlayer({ url, title, isCompleted, isLocked, onComplete }: VideoPlayerProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [showCompleteButton, setShowCompleteButton] = useState(false);

  // Extract YouTube video ID from URL
  const getYouTubeVideoId = (url: string) => {
    const regExp = /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|&v=)([^#&?]*).*/;
    const match = url.match(regExp);
    return match && match[2].length === 11 ? match[2] : null;
  };

  const videoId = getYouTubeVideoId(url);

  // Simulate video completion tracking (in real app, use YouTube iframe API)
  useEffect(() => {
    if (isPlaying && !isCompleted) {
      // Show complete button after 10 seconds of "watching"
      const timer = setTimeout(() => {
        setShowCompleteButton(true);
      }, 10000);
      return () => clearTimeout(timer);
    }
  }, [isPlaying, isCompleted]);

  const handleMarkComplete = () => {
    onComplete();
    setShowCompleteButton(false);
    setIsPlaying(false);
  };

  if (isLocked) {
    return (
      <div className="bg-gray-100 border border-gray-200 rounded-lg p-4 opacity-60">
        <div className="flex items-center gap-3">
          <Lock className="w-5 h-5 text-gray-400" />
          <div className="flex-1">
            <p className="font-medium text-gray-600">{title}</p>
            <p className="text-sm text-gray-500">Complete previous videos to unlock</p>
          </div>
        </div>
      </div>
    );
  }

  if (isCompleted) {
    return (
      <div className="bg-green-50 border border-green-200 rounded-lg p-4">
        <div className="flex items-center gap-3">
          <CheckCircle className="w-5 h-5 text-green-600" />
          <div className="flex-1">
            <p className="font-medium text-gray-900">{title}</p>
            <p className="text-sm text-green-600">Completed ✓</p>
          </div>
          <a
            href={url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm text-blue-600 hover:text-blue-700 underline"
          >
            Rewatch
          </a>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white border border-gray-300 rounded-lg overflow-hidden">
      {!isPlaying ? (
        <div className="relative">
          {videoId ? (
            <img
              src={`https://img.youtube.com/vi/${videoId}/maxresdefault.jpg`}
              alt={title}
              className="w-full h-48 object-cover"
            />
          ) : (
            <div className="w-full h-48 bg-gradient-to-r from-blue-500 to-purple-500"></div>
          )}
          <button
            onClick={() => setIsPlaying(true)}
            className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-40 hover:bg-opacity-50 transition-all group"
          >
            <div className="bg-white rounded-full p-4 group-hover:scale-110 transition-transform">
              <Play className="w-8 h-8 text-blue-600" />
            </div>
          </button>
        </div>
      ) : (
        <div className="relative">
          {videoId ? (
            <iframe
              width="100%"
              height="315"
              src={`https://www.youtube.com/embed/${videoId}?autoplay=1`}
              title={title}
              frameBorder="0"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
              className="w-full"
            ></iframe>
          ) : (
            <div className="w-full h-48 bg-gray-200 flex items-center justify-center">
              <p className="text-gray-600">Video player not available for this URL</p>
            </div>
          )}
        </div>
      )}
      <div className="p-4">
        <p className="font-medium text-gray-900 mb-2">{title}</p>
        <div className="flex items-center gap-3">
          <a
            href={url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm text-blue-600 hover:text-blue-700 underline"
          >
            Watch on YouTube
          </a>
          {showCompleteButton && (
            <button
              onClick={handleMarkComplete}
              className="ml-auto px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors text-sm font-medium"
            >
              Mark as Complete
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
