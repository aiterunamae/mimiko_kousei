import streamlit as st

st.title("Test App")
st.write("If you can see this, Streamlit is working!")

try:
    import google.genai as genai
    st.success("google.genai imported successfully")
except ImportError as e:
    st.error(f"Failed to import google.genai: {e}")
    try:
        import google.generativeai as genai
        st.warning("Using old google.generativeai library")
    except ImportError as e2:
        st.error(f"Failed to import google.generativeai: {e2}")