import { useState, useEffect } from 'react';
import { CircleCheck, Circle, ChevronDown, ChevronUp, BookOpen, Video, FileText, Code, ExternalLink } from 'lucide-react';
import type { Day } from '../data/curriculum';
import { VideoPlayer } from './VideoPlayer';

interface DayCardProps {
  day: Day;
  dayId: string;
  isCompleted: boolean;
  completedResources: string[];
  markResourceCompleted: (dayId: string, resourceUrl: string) => void;
}

export function DayCard({ day, dayId, isCompleted, completedResources = [], markResourceCompleted }: DayCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  // Get all video resources
  const videoResources = day.resources.filter(r => r.type === 'video');
  const otherResources = day.resources.filter(r => r.type !== 'video');

  // Check if a resource is completed
  const isResourceCompleted = (resourceUrl: string) => {
    return completedResources.includes(resourceUrl);
  };

  // Check if day should be auto-completed
  useEffect(() => {
    if (videoResources.length > 0) {
      const allVideosCompleted = videoResources.every(video => 
        isResourceCompleted(video.url)
      );
      
      // Auto-complete day when all videos are watched
      if (allVideosCompleted && !isCompleted) {
        // This will be handled by the parent component
      }
    }
  }, [completedResources, videoResources, isCompleted]);

  const getIconForType = (type: string) => {
    switch (type) {
      case 'video': return <Video className="size-4" />;
      case 'article': return <FileText className="size-4" />;
      case 'tutorial': return <Code className="size-4" />;
      case 'documentation': return <BookOpen className="size-4" />;
      default: return <ExternalLink className="size-4" />;
    }
  };

  // Calculate progress
  const totalVideos = videoResources.length;
  const completedVideos = videoResources.filter(v => isResourceCompleted(v.url)).length;
  const progress = totalVideos > 0 ? (completedVideos / totalVideos) * 100 : 0;

  return (
    <div className={`bg-white rounded-xl border-2 transition-all ${ 
      isCompleted ? 'border-green-500 shadow-lg' : 'border-gray-200 hover:border-gray-300'
    }`}>
      <div className="p-6">
        <div className="flex items-start gap-4">
          <div className="mt-1 flex-shrink-0">
            {isCompleted ? (
              <CircleCheck className="size-6 text-green-500" />
            ) : (
              <Circle className="size-6 text-gray-400" />
            )}
          </div>

          <div className="flex-1">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-xl font-bold text-gray-900">
                  Day {day.day}: {day.title}
                </h3>
                <p className="text-gray-600 mt-1">{day.description}</p>
                {totalVideos > 0 && (
                  <div className="mt-2">
                    <div className="flex items-center gap-2 text-sm">
                      <span className="text-gray-600">
                        Videos: {completedVideos}/{totalVideos}
                      </span>
                      <div className="flex-1 max-w-xs">
                        <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all duration-500"
                            style={{ width: `${progress}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
              <button
                onClick={() => setIsExpanded(!isExpanded)}
                className="ml-4 p-2 hover:bg-gray-100 rounded-lg transition-colors"
              >
                {isExpanded ? (
                  <ChevronUp className="size-5 text-gray-600" />
                ) : (
                  <ChevronDown className="size-5 text-gray-600" />
                )}
              </button>
            </div>

            <div className="flex flex-wrap gap-2 mt-3">
              {day.topics.map((topic, idx) => (
                <span
                  key={idx}
                  className="px-3 py-1 bg-gradient-to-r from-blue-50 to-purple-50 text-blue-700 rounded-full text-sm font-medium"
                >
                  {topic}
                </span>
              ))}
            </div>

            {isExpanded && (
              <div className="mt-6 space-y-6 pt-6 border-t border-gray-200">
                {/* Learning Objectives */}
                <div>
                  <h4 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
                    <BookOpen className="size-5 text-purple-600" />
                    Learning Objectives
                  </h4>
                  <ul className="space-y-2">
                    {day.objectives.map((objective, idx) => (
                      <li key={idx} className="flex items-start gap-2 text-gray-700">
                        <span className="text-purple-600 mt-1">•</span>
                        <span>{objective}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Video Resources - Sequential */}
                {videoResources.length > 0 && (
                  <div>
                    <h4 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
                      <Video className="size-5 text-blue-600" />
                      Video Tutorials (Watch in Order)
                    </h4>
                    <div className="space-y-4">
                      {videoResources.map((resource, idx) => {
                        const isCompleted = isResourceCompleted(resource.url);
                        const isLocked = idx > 0 && !isResourceCompleted(videoResources[idx - 1].url);
                        
                        return (
                          <VideoPlayer
                            key={idx}
                            url={resource.url}
                            title={`${idx + 1}. ${resource.title}`}
                            isCompleted={isCompleted}
                            isLocked={isLocked}
                            onComplete={() => {
                              // Pass total videos count for auto-completion
                              markResourceCompleted(dayId, resource.url);
                            }}
                          />
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* Other Resources */}
                {otherResources.length > 0 && (
                  <div>
                    <h4 className="font-semibold text-gray-900 mb-3">
                      Additional Resources
                    </h4>
                    <div className="space-y-2">
                      {otherResources.map((resource, idx) => (
                        <a
                          key={idx}
                          href={resource.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="flex items-center gap-3 p-3 bg-gray-50 hover:bg-gray-100 rounded-lg transition-colors group"
                        >
                          <div className="p-2 bg-white rounded-lg group-hover:bg-blue-50 transition-colors">
                            {getIconForType(resource.type)}
                          </div>
                          <div className="flex-1">
                            <p className="font-medium text-gray-900 group-hover:text-blue-600">
                              {resource.title}
                            </p>
                            <p className="text-sm text-gray-600">{resource.source}</p>
                          </div>
                          <ExternalLink className="size-4 text-gray-400 group-hover:text-blue-600" />
                        </a>
                      ))}
                    </div>
                  </div>
                )}

                {/* Practice Tasks */}
                {day.practice && (
                  <div>
                    <h4 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
                      <Code className="size-5 text-green-600" />
                      Practice & Exercises
                    </h4>
                    <ul className="space-y-2">
                      {day.practice.map((task, idx) => (
                        <li key={idx} className="flex items-start gap-2 text-gray-700">
                          <span className="text-green-600 mt-1">✓</span>
                          <span>{task}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Time Estimate */}
                <div className="bg-gradient-to-r from-blue-50 to-purple-50 p-4 rounded-lg">
                  <p className="text-sm text-gray-700">
                    <span className="font-semibold">Estimated Time:</span> {day.timeEstimate}
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}