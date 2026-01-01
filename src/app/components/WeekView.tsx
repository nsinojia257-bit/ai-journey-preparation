import { useEffect } from 'react';
import { curriculum } from '../data/curriculum';
import { DayCard } from './DayCard';

interface WeekViewProps {
  month: number;
  week: number;
  completedDays: Set<string>;
  completedResources: Record<string, string[]>;
  markResourceCompleted: (dayId: string, resourceUrl: string) => void;
}

export function WeekView({ month, week, completedDays, completedResources, markResourceCompleted }: WeekViewProps) {
  const monthData = curriculum.find(m => m.month === month);
  const weekData = monthData?.weeks.find(w => w.week === week);

  if (!weekData) return null;

  return (
    <div className="space-y-6">
      <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 border border-gray-200 shadow-sm">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">
              Month {month}, Week {week}
            </h2>
            <p className="text-gray-600 mt-1">{weekData.title}</p>
          </div>
          <div className="text-right">
            <p className="text-sm text-gray-600">Duration</p>
            <p className="font-semibold text-gray-900">{weekData.days.length} Days</p>
          </div>
        </div>
      </div>

      <div className="space-y-4">
        {weekData.days.map((day) => {
          const dayId = `${month}-${week}-${day.day}`;
          const isCompleted = completedDays.has(dayId);
          const dayCompletedResources = completedResources[dayId] || [];
          
          return (
            <DayCard
              key={dayId}
              day={day}
              dayId={dayId}
              isCompleted={isCompleted}
              completedResources={dayCompletedResources}
              markResourceCompleted={markResourceCompleted}
            />
          );
        })}
      </div>
    </div>
  );
}