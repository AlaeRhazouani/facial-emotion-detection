import streamlit as st


def render_confidence_bar(scores: dict, predicted_emotion: str):
    """
    Renders confidence scores as progress bars.
    scores            → {emotion: probability} dict
    predicted_emotion → highlighted with checkmark
    """
    st.subheader("Confidence Scores")

    # Sort by confidence descending
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    for emotion, score in sorted_scores:
        label = emotion.capitalize()

        if emotion == predicted_emotion:
            st.markdown(f"**{label} (predicted)**")
        else:
            st.markdown(f"{label}")

        st.progress(int(score))
        st.caption(f"{score:.1f}%")