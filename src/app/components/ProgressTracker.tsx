import { Target, TrendingUp, Calendar } from 'lucide-react';
import { curriculum } from '../data/curriculum';

interface ProgressTrackerProps {
  selectedMonth: number;
  completedDays: Set<string>;
}

export function ProgressTracker({ selectedMonth, completedDays }: ProgressTrackerProps) {
  const totalDays = curriculum.reduce((acc, month) => 
    acc + month.weeks.reduce((weekAcc, week) => weekAcc + week.days.length, 0), 0
  );
  
  const completedCount = completedDays.size;
  const overallProgress = Math.round((completedCount / totalDays) * 100);

  const currentMonth = curriculum.find(m => m.month === selectedMonth);
  const monthTotalDays = currentMonth?.weeks.reduce((acc, week) => acc + week.days.length, 0) || 0;
  const monthCompleted = Array.from(completedDays).filter(dayId => 
    dayId.startsWith(`month-${selectedMonth}-`)
  ).length;
  const monthProgress = Math.round((monthCompleted / monthTotalDays) * 100);

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
      <div className="bg-white/80 backdrop-blur-sm rounded-xl p-6 border border-gray-200 shadow-sm">
        <div className="flex items-center gap-3 mb-3">
          <div className="p-2 bg-blue-100 rounded-lg">
            <Target className="size-5 text-blue-600" />
          </div>
          <h3 className="font-semibold text-gray-900">Overall Progress</h3>
        </div>
        <div className="flex items-end gap-2">
          <span className="text-3xl font-bold text-gray-900">{overallProgress}%</span>
          <span className="text-gray-600 mb-1">({completedCount}/{totalDays} days)</span>
        </div>
        <div className="mt-3 bg-gray-200 rounded-full h-2 overflow-hidden">
          <div 
            className="h-full bg-gradient-to-r from-blue-500 to-purple-600 transition-all duration-500"
            style={{ width: `${overallProgress}%` }}
          />
        </div>
      </div>

      <div className="bg-white/80 backdrop-blur-sm rounded-xl p-6 border border-gray-200 shadow-sm">
        <div className="flex items-center gap-3 mb-3">
          <div className="p-2 bg-purple-100 rounded-lg">
            <TrendingUp className="size-5 text-purple-600" />
          </div>
          <h3 className="font-semibold text-gray-900">Current Month</h3>
        </div>
        <div className="flex items-end gap-2">
          <span className="text-3xl font-bold text-gray-900">{monthProgress}%</span>
          <span className="text-gray-600 mb-1">({monthCompleted}/{monthTotalDays} days)</span>
        </div>
        <div className="mt-3 bg-gray-200 rounded-full h-2 overflow-hidden">
          <div 
            className="h-full bg-gradient-to-r from-purple-500 to-pink-600 transition-all duration-500"
            style={{ width: `${monthProgress}%` }}
          />
        </div>
      </div>

      <div className="bg-white/80 backdrop-blur-sm rounded-xl p-6 border border-gray-200 shadow-sm">
        <div className="flex items-center gap-3 mb-3">
          <div className="p-2 bg-green-100 rounded-lg">
            <Calendar className="size-5 text-green-600" />
          </div>
          <h3 className="font-semibold text-gray-900">Time Remaining</h3>
        </div>
        <div className="flex items-end gap-2">
          <span className="text-3xl font-bold text-gray-900">{totalDays - completedCount}</span>
          <span className="text-gray-600 mb-1">days left</span>
        </div>
        <p className="text-sm text-gray-600 mt-3">
          {Math.ceil((totalDays - completedCount) / 7)} weeks to completion
        </p>
      </div>
    </div>
  );
}
