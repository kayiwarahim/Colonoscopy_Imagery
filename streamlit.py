import importlib.util
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent

# Prevent this launcher file from shadowing the installed `streamlit` package.
sys.path = [path for path in sys.path if Path(path).resolve() != PROJECT_ROOT]
sys.path.append(str(PROJECT_ROOT))
app_module_path = PROJECT_ROOT / "streamlit_app.py"
spec = importlib.util.spec_from_file_location("streamlit_app", app_module_path)
streamlit_app = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(streamlit_app)


if __name__ == "__main__":
    streamlit_app.main()
