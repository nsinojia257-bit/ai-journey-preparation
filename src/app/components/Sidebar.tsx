import { BookOpen, Calendar, CircleCheck } from 'lucide-react';
import { curriculum } from '../data/curriculum';

interface SidebarProps {
  selectedMonth: number;
  setSelectedMonth: (month: number) => void;
  setSelectedWeek: (week: number) => void;
  completedDays: Set<string>;
}

export function Sidebar({ selectedMonth, setSelectedMonth, setSelectedWeek, completedDays }: SidebarProps) {
  const calculateMonthProgress = (monthNum: number) => {
    const monthData = curriculum.find(m => m.month === monthNum);
    if (!monthData) return 0;
    
    const totalDays = monthData.weeks.reduce((acc, week) => acc + week.days.length, 0);
    const completedInMonth = Array.from(completedDays).filter(dayId => 
      dayId.startsWith(`month-${monthNum}-`)
    ).length;
    
    return Math.round((completedInMonth / totalDays) * 100);
  };

  return (
    <aside className="fixed left-0 top-0 h-screen w-80 bg-white/90 backdrop-blur-sm border-r border-gray-200 overflow-y-auto">
      <div className="p-6">
        <div className="flex items-center gap-3 mb-8">
          <div className="p-2 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg">
            <BookOpen className="size-6 text-white" />
          </div>
          <div>
            <h2 className="font-bold text-gray-900">AI Learning Path</h2>
            <p className="text-sm text-gray-600">5 Month Journey</p>
          </div>
        </div>

        <nav className="space-y-2">
          {curriculum.map((month) => {
            const progress = calculateMonthProgress(month.month);
            const isSelected = selectedMonth === month.month;
            
            return (
              <div key={month.month} className="space-y-1">
                <button
                  onClick={() => {
                    setSelectedMonth(month.month);
                    setSelectedWeek(1);
                  }}
                  className={`w-full text-left px-4 py-3 rounded-lg transition-all ${
                    isSelected 
                      ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-lg' 
                      : 'hover:bg-gray-100 text-gray-700'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-semibold">Month {month.month}</span>
                    {progress === 100 && (
                      <CircleCheck className="size-5 text-green-400" />
                    )}
                  </div>
                  <p className={`text-sm ${isSelected ? 'text-blue-100' : 'text-gray-600'}`}>
                    {month.title}
                  </p>
                  <div className="mt-2 bg-white/20 rounded-full h-1.5 overflow-hidden">
                    <div 
                      className={`h-full transition-all ${isSelected ? 'bg-white' : 'bg-blue-500'}`}
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                </button>
                
                {isSelected && (
                  <div className="ml-4 space-y-1 mt-2">
                    {month.weeks.map((week) => (
                      <button
                        key={week.week}
                        onClick={() => setSelectedWeek(week.week)}
                        className="w-full text-left px-3 py-2 rounded-md hover:bg-gray-100 text-sm text-gray-700 flex items-center gap-2"
                      >
                        <Calendar className="size-4" />
                        Week {week.week}: {week.title}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </nav>
      </div>
    </aside>
  );
}
