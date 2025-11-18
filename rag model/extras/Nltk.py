import nltk
import ssl

try:
    # Bypass SSL verification if needed
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # Download required NLTK data
    print("Downloading NLTK data...")
    nltk.download('punkt_tab')
    nltk.download('punkt')
    print("NLTK data downloaded successfully!")
    
except Exception as e:
    print(f"Error downloading NLTK data: {e}")
    print("The code will use simple splitting instead.")