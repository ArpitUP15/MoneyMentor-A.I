import os
import sqlalchemy
from sqlalchemy import text

# Print the database URL (obscuring the password)
db_url = os.getenv('DATABASE_URL', 'No DATABASE_URL found')
if 'postgres://' in db_url:
    # Replace password with ***
    parts = db_url.split('@')
    if len(parts) >= 2:
        user_pass = parts[0].split('://')
        if len(user_pass) >= 2:
            scheme = user_pass[0]
            user_pass_parts = user_pass[1].split(':')
            if len(user_pass_parts) >= 2:
                user = user_pass_parts[0]
                safe_url = f"{scheme}://{user}:***@{parts[1]}"
                print(f"Using database URL: {safe_url}")
else:
    print(f"Using database URL: {db_url}")

try:
    # Create engine
    engine = sqlalchemy.create_engine(db_url)
    
    # Test connection
    with engine.connect() as connection:
        # Execute a test query
        result = connection.execute(text("SELECT 1"))
        
        # Fetch result
        data = result.fetchone()
        
        # Print result
        print(f"Database connection successful! Test query result: {data}")
        
        # Check version
        result = connection.execute(text("SELECT version()"))
        version = result.fetchone()[0]
        print(f"PostgreSQL version: {version}")
        
except Exception as e:
    print(f"Error connecting to database: {str(e)}")