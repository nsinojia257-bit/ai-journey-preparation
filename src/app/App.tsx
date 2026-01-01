import { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { WeekView } from './components/WeekView';
import { ProgressTracker } from './components/ProgressTracker';
import { AuthPage } from './components/AuthPage';
import { LogOut, User } from 'lucide-react';
import { projectId, publicAnonKey } from '../../utils/supabase/info';
import { supabase } from './lib/supabase';
import { curriculum } from './data/curriculum';

interface UserProgress {
  completedDays: string[];
  completedResources: Record<string, string[]>;
}

export default function App() {
  const [selectedMonth, setSelectedMonth] = useState(1);
  const [selectedWeek, setSelectedWeek] = useState(1);
  const [userProgress, setUserProgress] = useState<UserProgress>({
    completedDays: [],
    completedResources: {}
  });
  const [accessToken, setAccessToken] = useState<string | null>(null);
  const [userName, setUserName] = useState<string>('');
  const [loading, setLoading] = useState(true);

  // Check for existing session on mount
  useEffect(() => {
    const checkSession = async () => {
      try {
        const { data: { session } } = await supabase.auth.getSession();
        
        if (session?.access_token) {
          setAccessToken(session.access_token);
          setUserName(session.user?.user_metadata?.name || session.user?.email?.split('@')[0] || 'User');
          await loadUserProgress(session.access_token);
        }
      } catch (error) {
        console.error('Session check error:', error);
      } finally {
        setLoading(false);
      }
    };

    checkSession();
  }, []);

  // Load user progress from backend
  const loadUserProgress = async (token: string) => {
    try {
      const response = await fetch(
        `https://${projectId}.supabase.co/functions/v1/make-server-251554e1/progress`,
        {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        }
      );

      if (response.ok) {
        const data = await response.json();
        setUserProgress(data);
      } else {
        console.error('Failed to load progress:', await response.text());
      }
    } catch (error) {
      console.error('Error loading progress:', error);
    }
  };

  // Save user progress to backend
  const saveUserProgress = async (token: string, progress: UserProgress) => {
    try {
      const response = await fetch(
        `https://${projectId}.supabase.co/functions/v1/make-server-251554e1/progress`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
          },
          body: JSON.stringify(progress)
        }
      );

      if (!response.ok) {
        console.error('Failed to save progress:', await response.text());
      }
    } catch (error) {
      console.error('Error saving progress:', error);
    }
  };

  // Mark a resource (video) as completed
  const markResourceCompleted = async (dayId: string, resourceUrl: string) => {
    if (!accessToken) return;

    // Get the day's data to count total videos
    const [month, week, day] = dayId.split('-').map(Number);
    const monthData = curriculum.find(m => m.month === month);
    const weekData = monthData?.weeks.find(w => w.week === week);
    const dayData = weekData?.days.find(d => d.day === day);
    const totalVideos = dayData?.resources.filter(r => r.type === 'video').length || 0;

    try {
      const response = await fetch(
        `https://${projectId}.supabase.co/functions/v1/make-server-251554e1/progress/resource`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${accessToken}`
          },
          body: JSON.stringify({ dayId, resourceUrl, totalVideos })
        }
      );

      if (response.ok) {
        const data = await response.json();
        setUserProgress(data.progress);
      } else {
        console.error('Failed to mark resource complete:', await response.text());
      }
    } catch (error) {
      console.error('Error marking resource complete:', error);
    }
  };

  const handleAuthSuccess = async (token: string, name: string) => {
    setAccessToken(token);
    setUserName(name);
    await loadUserProgress(token);
  };

  const handleLogout = async () => {
    try {
      await supabase.auth.signOut();
      setAccessToken(null);
      setUserName('');
      setUserProgress({
        completedDays: [],
        completedResources: {}
      });
    } catch (error) {
      console.error('Logout error:', error);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-600 mx-auto"></div>
          <p className="text-gray-600 mt-4">Loading...</p>
        </div>
      </div>
    );
  }

  if (!accessToken) {
    return <AuthPage onAuthSuccess={handleAuthSuccess} />;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50">
      <div className="flex">
        <Sidebar 
          selectedMonth={selectedMonth}
          setSelectedMonth={setSelectedMonth}
          setSelectedWeek={setSelectedWeek}
          completedDays={new Set(userProgress.completedDays)}
        />
        <div className="flex-1 ml-80">
          <header className="bg-white/80 backdrop-blur-sm border-b border-gray-200 sticky top-0 z-10">
            <div className="max-w-7xl mx-auto px-8 py-6">
              <div className="flex items-center justify-between">
                <div>
                  <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 bg-clip-text text-transparent">
                    AI Mastery: Zero to Expert in 5 Months
                  </h1>
                  <p className="text-gray-600 mt-2">
                    A comprehensive day-by-day roadmap to become an AI expert
                  </p>
                </div>
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-blue-100 to-purple-100 rounded-lg">
                    <User className="w-5 h-5 text-purple-600" />
                    <span className="font-medium text-gray-700">{userName}</span>
                  </div>
                  <button
                    onClick={handleLogout}
                    className="flex items-center gap-2 px-4 py-2 bg-red-100 hover:bg-red-200 text-red-700 rounded-lg transition-colors"
                  >
                    <LogOut className="w-5 h-5" />
                    Logout
                  </button>
                </div>
              </div>
            </div>
          </header>
          
          <main className="max-w-7xl mx-auto px-8 py-8">
            <ProgressTracker 
              selectedMonth={selectedMonth}
              completedDays={new Set(userProgress.completedDays)}
            />
            
            <WeekView 
              month={selectedMonth}
              week={selectedWeek}
              completedDays={new Set(userProgress.completedDays)}
              completedResources={userProgress.completedResources}
              markResourceCompleted={markResourceCompleted}
            />
          </main>
        </div>
      </div>
    </div>
  );
}