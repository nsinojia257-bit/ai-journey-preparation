import { Hono } from "npm:hono";
import { cors } from "npm:hono/cors";
import { logger } from "npm:hono/logger";
import * as kv from "./kv_store.tsx";
import { createClient } from "npm:@supabase/supabase-js@2";

const app = new Hono();

// Enable logger
app.use('*', logger(console.log));

// Enable CORS for all routes and methods
app.use(
  "/*",
  cors({
    origin: "*",
    allowHeaders: ["Content-Type", "Authorization"],
    allowMethods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    exposeHeaders: ["Content-Length"],
    maxAge: 600,
  }),
);

// Health check endpoint
app.get("/make-server-251554e1/health", (c) => {
  return c.json({ status: "ok" });
});

// Sign up endpoint
app.post("/make-server-251554e1/signup", async (c) => {
  try {
    const { email, password, name } = await c.req.json();
    
    if (!email || !password) {
      return c.json({ error: "Email and password are required" }, 400);
    }

    const supabase = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? '',
    );

    const { data, error } = await supabase.auth.admin.createUser({
      email,
      password,
      user_metadata: { name: name || email.split('@')[0] },
      // Automatically confirm the user's email since an email server hasn't been configured.
      email_confirm: true
    });

    if (error) {
      console.log(`Sign up error for ${email}: ${error.message}`);
      return c.json({ error: error.message }, 400);
    }

    return c.json({ 
      success: true, 
      user: { 
        id: data.user.id, 
        email: data.user.email,
        name: data.user.user_metadata?.name 
      } 
    });
  } catch (error) {
    console.log(`Sign up exception: ${error}`);
    return c.json({ error: "Sign up failed" }, 500);
  }
});

// Get user progress
app.get("/make-server-251554e1/progress", async (c) => {
  try {
    // Use ANON_KEY to validate user JWT tokens
    const supabase = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_ANON_KEY') ?? '',
    );

    const accessToken = c.req.header('Authorization')?.split(' ')[1];
    if (!accessToken) {
      return c.json({ error: "Unauthorized - no token provided" }, 401);
    }

    const { data: { user }, error } = await supabase.auth.getUser(accessToken);
    if (error || !user?.id) {
      console.log(`Authorization error while getting user progress: ${error?.message}`);
      return c.json({ error: "Unauthorized - invalid token" }, 401);
    }

    const progressKey = `user_progress_${user.id}`;
    const progress = await kv.get(progressKey);
    
    // Parse progress or return default structure
    const userProgress = progress ? JSON.parse(progress) : {
      completedDays: [],
      completedResources: {}
    };
    
    return c.json(userProgress);
  } catch (error) {
    console.log(`Error fetching user progress: ${error}`);
    return c.json({ error: "Failed to fetch progress" }, 500);
  }
});

// Save user progress
app.post("/make-server-251554e1/progress", async (c) => {
  try {
    // Use ANON_KEY to validate user JWT tokens
    const supabase = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_ANON_KEY') ?? '',
    );

    const accessToken = c.req.header('Authorization')?.split(' ')[1];
    if (!accessToken) {
      return c.json({ error: "Unauthorized - no token provided" }, 401);
    }

    const { data: { user }, error } = await supabase.auth.getUser(accessToken);
    if (error || !user?.id) {
      console.log(`Authorization error while saving user progress: ${error?.message}`);
      return c.json({ error: "Unauthorized - invalid token" }, 401);
    }

    const progressData = await c.req.json();
    
    // Validate the structure
    if (!progressData.completedDays || !Array.isArray(progressData.completedDays)) {
      return c.json({ error: "completedDays must be an array" }, 400);
    }

    if (!progressData.completedResources || typeof progressData.completedResources !== 'object') {
      return c.json({ error: "completedResources must be an object" }, 400);
    }

    const progressKey = `user_progress_${user.id}`;
    await kv.set(progressKey, JSON.stringify(progressData));
    
    return c.json({ success: true });
  } catch (error) {
    console.log(`Error saving user progress: ${error}`);
    return c.json({ error: "Failed to save progress" }, 500);
  }
});

// Mark a resource as completed
app.post("/make-server-251554e1/progress/resource", async (c) => {
  try {
    const supabase = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_ANON_KEY') ?? '',
    );

    const accessToken = c.req.header('Authorization')?.split(' ')[1];
    if (!accessToken) {
      return c.json({ error: "Unauthorized - no token provided" }, 401);
    }

    const { data: { user }, error } = await supabase.auth.getUser(accessToken);
    if (error || !user?.id) {
      console.log(`Authorization error while marking resource complete: ${error?.message}`);
      return c.json({ error: "Unauthorized - invalid token" }, 401);
    }

    const { dayId, resourceUrl, totalVideos } = await c.req.json();
    
    if (!dayId || !resourceUrl) {
      return c.json({ error: "dayId and resourceUrl are required" }, 400);
    }

    const progressKey = `user_progress_${user.id}`;
    const progress = await kv.get(progressKey);
    const userProgress = progress ? JSON.parse(progress) : {
      completedDays: [],
      completedResources: {}
    };

    // Initialize completedResources for this day if it doesn't exist
    if (!userProgress.completedResources[dayId]) {
      userProgress.completedResources[dayId] = [];
    }

    // Add resource if not already completed
    if (!userProgress.completedResources[dayId].includes(resourceUrl)) {
      userProgress.completedResources[dayId].push(resourceUrl);
    }

    // Auto-complete day if all videos are watched
    if (totalVideos && userProgress.completedResources[dayId].length >= totalVideos) {
      if (!userProgress.completedDays.includes(dayId)) {
        userProgress.completedDays.push(dayId);
      }
    }

    await kv.set(progressKey, JSON.stringify(userProgress));
    
    return c.json({ success: true, progress: userProgress });
  } catch (error) {
    console.log(`Error marking resource complete: ${error}`);
    return c.json({ error: "Failed to mark resource complete" }, 500);
  }
});

Deno.serve(app.fetch);