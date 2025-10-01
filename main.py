import uvicorn
from fastapi import FastAPI

# 1. Create an instance of the FastAPI class
app = FastAPI()

# 2. Define a route for '/hello'
@app.get("/hello")
async def read_hello():
  """This function runs when someone accesses the /hello endpoint."""
  return {"message": "hello world"}

# 3. Add a main block to run the app with uvicorn
#    This makes the script runnable with "python main.py"
if __name__ == "__main__":
  uvicorn.run(app, host="127.0.0.1", port=8000)