import os
import datetime
import csv
import json
from dotenv import load_dotenv
from supabase import create_client, Client

class DBManager:
    def __init__(self, db_path=None):
        # db_path parameter is kept for backward compatibility but is no longer used
        self.db_path = db_path
        
        # Load environment variables
        load_dotenv()
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        
        if not url or not key:
            print("Warning: SUPABASE_URL or SUPABASE_KEY not found in environment. DB connection may fail.")
            self.supabase = None
        else:
            self.supabase: Client = create_client(url, key)

    def init_db(self):
        """No-op. Supabase handles table schema remotely."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def log_session(self, exercise_type, reps, duration, knee_cave=0, shoulder_sway=0):
        """Log a completed session to Supabase."""
        if not self.supabase:
            return None
            
        data = {
            "exercise_type": exercise_type,
            "reps": reps,
            "duration_seconds": duration,
            "knee_cave_incidents": knee_cave,
            "shoulder_sway_incidents": shoulder_sway
        }
        
        try:
            response = self.supabase.table("sessions").insert(data).execute()
            if response.data and len(response.data) > 0:
                # Return the newly created id
                return response.data[0].get("id")
        except Exception as e:
            print(f"Error logging session: {e}")
            
        return None

    def get_total_reps(self, exercise_type):
        if not self.supabase:
            return 0
            
        try:
            response = self.supabase.table("sessions").select("reps").eq("exercise_type", exercise_type).execute()
            if response.data:
                return sum(row.get("reps", 0) for row in response.data)
            return 0
        except Exception as e:
            print(f"Error getting total reps: {e}")
            return 0
            
    def get_recent_sessions(self, limit=10):
        """Retrieve the most recent sessions."""
        if not self.supabase:
            return []
            
        try:
            response = self.supabase.table("sessions").select("*").order("id", desc=True).limit(limit).execute()
            return response.data
        except Exception as e:
            print(f"Error getting recent sessions: {e}")
            return []

    def export_session_summary(self, session_id=None, format='csv'):
        """Export session summary to a file."""
        if not self.supabase:
            return None
            
        try:
            if session_id:
                response = self.supabase.table("sessions").select("*").eq("id", session_id).execute()
            else:
                response = self.supabase.table("sessions").select("*").order("id", desc=True).limit(1).execute()
                
            rows = response.data
            if not rows:
                print("No sessions to export.")
                return

            filename = f"session_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
            headers = list(rows[0].keys())
            
            if format == 'csv':
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
                    writer.writerows([[row.get(h) for h in headers] for row in rows])
            elif format == 'json':
                with open(filename, 'w') as f:
                    json.dump(rows, f, indent=4)
                    
            print(f"Summary exported to {filename}")
        except Exception as e:
            print(f"Export failed: {e}")

