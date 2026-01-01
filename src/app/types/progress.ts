// Track which resources (videos) have been completed by users
export interface UserProgress {
  completedDays: string[]; // Array of day IDs like "1-1-1" (month-week-day)
  completedResources: Record<string, string[]>; // dayId -> array of resource URLs that are completed
}

export interface VideoProgress {
  dayId: string;
  resourceUrl: string;
  completed: boolean;
  timestamp: number;
}
