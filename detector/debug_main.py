import debugpy
debugpy.listen(("0.0.0.0", 5678))
print("⏳ Waiting for debugger attach on port 5678...")
debugpy.wait_for_client()
print("🔗 Debugger attached, starting worker...")
from detector.main import main
main()