import sqlite3

def generate_report():
    print("=" * 50)
    print("Form Verification Test Report")
    print("=" * 50)
    
    try:
        with sqlite3.connect("data/fitness_data.db") as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Check if columns exist
            cursor.execute("PRAGMA table_info(sessions)")
            columns = [col['name'] for col in cursor.fetchall()]
            
            if 'knee_cave_incidents' not in columns or 'shoulder_sway_incidents' not in columns:
                print("❌ ERROR: Database migration failed. Biomechanical columns are missing.")
                return
            else:
                print("✅ Schema Verified: Biomechanical columns exist in 'sessions'.")
            
            # Fetch recent sessions
            cursor.execute("SELECT id, exercise_type, reps, knee_cave_incidents, shoulder_sway_incidents FROM sessions ORDER BY id DESC LIMIT 5")
            rows = cursor.fetchall()
            
            if not rows:
                print("⚠️ No session data found. Run the app and complete some reps.")
                return
                
            print("\nRecent Session Logs:")
            for row in rows:
                print(f"ID {row['id']} | {row['exercise_type']} | Reps: {row['reps']} | "
                      f"Knee Cave: {row['knee_cave_incidents']} | Shoulder Sway: {row['shoulder_sway_incidents']}")
                      
            print("\n✅ Verification Successful. Error states are actively being logged.")
            print("=" * 50)
            
    except Exception as e:
        print(f"❌ ERROR connecting to database: {e}")

if __name__ == "__main__":
    generate_report()
