import streamlit as st
import torch
from gliner import GLiNER
import time
from hotel_dataset import HOTEL_DATASET, HOTEL_LABELS

@st.cache_resource
def load_model():
    """Load the GLiNER model and cache it"""
    with st.spinner("Loading GLiNER model..."):
        model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
    return model

def highlight_entities(text, entities):
    """Highlight entities in the text with different colors"""
    if not entities:
        return text
    
    # Sort entities by start position in reverse order to avoid position shifts
    sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
    
    # Color palette for different entity types
    colors = [
        "#FFB6C1", "#87CEEB", "#98FB98", "#F0E68C", "#DDA0DD",
        "#FFE4B5", "#AFEEEE", "#D3D3D3", "#FFA07A", "#20B2AA"
    ]
    
    # Get unique labels and assign colors
    unique_labels = list(set([entity['label'] for entity in entities]))
    label_colors = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
    
    highlighted_text = text
    
    for entity in sorted_entities:
        start, end = entity['start'], entity['end']
        entity_text = entity['text']
        label = entity['label']
        color = label_colors[label]
        
        # Create highlighted span
        highlighted_span = f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px; margin: 1px;">{entity_text} <sup style="font-size: 10px;">{label}</sup></span>'
        
        # Replace the entity in the text
        highlighted_text = highlighted_text[:start] + highlighted_span + highlighted_text[end:]
    
    return highlighted_text

def calculate_metrics(predicted_entities, ground_truth_entities):
    """Calculate Precision, Recall, and F1 score for entity recognition"""
    # Convert entities to sets of (text, label, start, end) tuples for comparison
    pred_set = set()
    for entity in predicted_entities:
        pred_set.add((entity['text'], entity['label'], entity['start'], entity['end']))
    
    gt_set = set()
    for entity in ground_truth_entities:
        gt_set.add((entity['text'], entity['label'], entity['start'], entity['end']))
    
    # Calculate metrics
    true_positives = len(pred_set.intersection(gt_set))
    false_positives = len(pred_set - gt_set)
    false_negatives = len(gt_set - pred_set)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }

def evaluate_dataset(model, dataset, labels, confidence_threshold=0.5):
    """Evaluate the model on the hotel dataset"""
    all_metrics = []
    all_predictions = []
    
    for sample in dataset:
        text = sample['text']
        ground_truth = sample['entities']
        
        # Predict entities
        predicted = model.predict_entities(text, labels)
        
        # Store raw predictions for later use
        all_predictions.append(predicted)
        
        # Filter by confidence threshold
        predicted_filtered = [
            entity for entity in predicted 
            if entity.get('score', 1.0) >= confidence_threshold
        ]
        
        # Calculate metrics for this sample
        metrics = calculate_metrics(predicted_filtered, ground_truth)
        all_metrics.append(metrics)
    
    # Calculate average metrics
    avg_precision = sum(m['precision'] for m in all_metrics) / len(all_metrics)
    avg_recall = sum(m['recall'] for m in all_metrics) / len(all_metrics)
    avg_f1 = sum(m['f1'] for m in all_metrics) / len(all_metrics)
    
    total_tp = sum(m['true_positives'] for m in all_metrics)
    total_fp = sum(m['false_positives'] for m in all_metrics)
    total_fn = sum(m['false_negatives'] for m in all_metrics)
    
    return {
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_f1': avg_f1,
        'total_true_positives': total_tp,
        'total_false_positives': total_fp,
        'total_false_negatives': total_fn,
        'sample_metrics': all_metrics,
        'all_predictions': all_predictions
    }

def main():
    st.set_page_config(
        page_title="GLiNER Hotel NER Demo",
        page_icon="🏨",
        layout="wide"
    )
    
    st.title("🏨 GLiNER Hotel Named Entity Recognition Demo")
    st.markdown("**GLiNER** is a Named Entity Recognition model that can identify entities based on custom labels. This demo focuses on hotel entity recognition.")
    
    # Load model
    try:
        model = load_model()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()
    
    # Mode selection
    mode = st.radio(
        "Select Mode:",
        ["🔍 Single Text Analysis", "📊 Hotel Dataset Evaluation"],
        horizontal=True
    )
    
    if mode == "🔍 Single Text Analysis":
        # Single text analysis mode
        default_text = """The Grand Palace Hotel is located in the heart of downtown Manhattan, New York. This luxurious 5-star hotel offers 200 elegantly appointed rooms and suites with stunning city views. Guests can enjoy fine dining at our award-winning restaurant Le Bernardin, relax at the rooftop spa, or take advantage of our 24-hour fitness center. Room rates start from $350 per night."""
        
        default_labels = "hotel, location, rooms, rating, restaurant, amenity, price"
        
        # Main interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📝 Input Text")
            text = st.text_area(
                "Enter the text you want to analyze:",
                value=default_text,
                height=200,
                help="Paste or type the text you want to extract entities from."
            )
        
        with col2:
            st.subheader("🏷️ Entity Labels")
            labels_input = st.text_area(
                "Enter comma-separated labels:",
                value=default_labels,
                height=100,
                help="Enter the types of entities you want to extract, separated by commas."
            )
            
            # Parse labels
            labels = [label.strip() for label in labels_input.split(",") if label.strip()]
            
            st.write("**Labels to extract:**")
            for i, label in enumerate(labels, 1):
                st.write(f"{i}. {label}")
            
            # Confidence threshold
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Only show entities with confidence above this threshold."
            )
        
        # Process button
        if st.button("🔍 Extract Entities", type="primary"):
            if not text.strip():
                st.warning("⚠️ Please enter some text to analyze.")
            elif not labels:
                st.warning("⚠️ Please enter at least one label.")
            else:
                with st.spinner("🔍 Extracting entities..."):
                    start_time = time.time()
                    
                    try:
                        # Predict entities
                        entities = model.predict_entities(text, labels)
                        
                        # Filter by confidence threshold
                        filtered_entities = [
                            entity for entity in entities 
                            if entity.get('score', 1.0) >= confidence_threshold
                        ]
                        
                        end_time = time.time()
                        processing_time = end_time - start_time
                        
                        # Display results
                        st.subheader("🎯 Results")
                        
                        if filtered_entities:
                            st.success(f"✅ Found {len(filtered_entities)} entities (processed in {processing_time:.2f}s)")
                            
                            # Show highlighted text
                            st.subheader("📄 Highlighted Text")
                            highlighted_text = highlight_entities(text, filtered_entities)
                            st.markdown(highlighted_text, unsafe_allow_html=True)
                            
                            # Show entities table
                            st.subheader("📊 Extracted Entities")
                            
                            entities_data = []
                            for entity in filtered_entities:
                                entities_data.append({
                                    "Entity": entity["text"],
                                    "Label": entity["label"],
                                    "Confidence": f"{entity.get('score', 1.0):.3f}",
                                    "Position": f"{entity['start']}-{entity['end']}"
                                })
                            
                            st.dataframe(entities_data, use_container_width=True)
                            
                            # Show statistics
                            st.subheader("📈 Statistics")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("Total Entities", len(filtered_entities))
                            
                            with col2:
                                st.metric("Unique Labels", len(set([e['label'] for e in filtered_entities])))
                        
                        else:
                            st.warning("⚠️ No entities found with the current labels and confidence threshold.")
                            st.info("💡 Try lowering the confidence threshold or using different labels.")
                    
                    except Exception as e:
                        st.error(f"❌ Error during prediction: {str(e)}")
    
    else:
        # Hotel dataset evaluation mode
        st.subheader("🏨 Hotel Dataset Evaluation")
        st.markdown(f"This dataset contains **{len(HOTEL_DATASET)} hotel descriptions** with ground truth annotations for hotel name recognition.")
        
        # Configuration for evaluation
        col1, col2 = st.columns(2)
        
        with col1:
            # Confidence threshold
            eval_confidence_threshold = st.slider(
                "Confidence Threshold for Evaluation",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Only consider predictions above this confidence threshold."
            )
        
        with col2:
            st.write("**Entity Labels in Dataset:**")
            for i, label in enumerate(HOTEL_LABELS, 1):
                st.write(f"{i}. {label}")
        
        # Evaluate button
        if st.button("📊 Evaluate on Hotel Dataset", type="primary"):
            with st.spinner("🔍 Evaluating model on hotel dataset..."):
                start_time = time.time()
                
                try:
                    # Evaluate the model
                    results = evaluate_dataset(model, HOTEL_DATASET, HOTEL_LABELS, eval_confidence_threshold)
                    
                    # Store results in session state for caching
                    st.session_state.evaluation_results = results
                    st.session_state.eval_confidence_threshold = eval_confidence_threshold
                    
                    end_time = time.time()
                    processing_time = end_time - start_time
                    
                    # Display evaluation results
                    st.subheader("📊 Evaluation Results")
                    st.success(f"✅ Evaluation completed in {processing_time:.2f}s")
                
                except Exception as e:
                    st.error(f"❌ Error during evaluation: {str(e)}")
        
        # Display results if they exist in session state
        if hasattr(st.session_state, 'evaluation_results'):
            results = st.session_state.evaluation_results
            cached_threshold = st.session_state.eval_confidence_threshold
            
            # Show results
            if not hasattr(st.session_state, 'just_evaluated'):
                st.subheader("📊 Evaluation Results")
                st.info("📋 Using cached evaluation results. Run evaluation again if you changed the confidence threshold.")
            
            # Overall metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Precision", f"{results['avg_precision']:.3f}")
            with col2:
                st.metric("Recall", f"{results['avg_recall']:.3f}")
            with col3:
                st.metric("F1 Score", f"{results['avg_f1']:.3f}")
            
            # Detailed metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("True Positives", results['total_true_positives'])
            with col2:
                st.metric("False Positives", results['total_false_positives'])
            with col3:
                st.metric("False Negatives", results['total_false_negatives'])
            
            # Sample-by-sample results
            st.subheader("📋 Sample-by-Sample Results")
            
            sample_results = []
            for i, (sample, metrics) in enumerate(zip(HOTEL_DATASET, results['sample_metrics'])):
                sample_results.append({
                    "Sample": i + 1,
                    "Hotel": sample['text'][:50] + "...",
                    "Precision": f"{metrics['precision']:.3f}",
                    "Recall": f"{metrics['recall']:.3f}",
                    "F1": f"{metrics['f1']:.3f}",
                    "TP": metrics['true_positives'],
                    "FP": metrics['false_positives'],
                    "FN": metrics['false_negatives']
                })
            
            st.dataframe(sample_results, use_container_width=True)
            
            # Show a detailed example
            st.subheader("🔍 Detailed Example")
            sample_idx = st.selectbox("Select sample to view:", range(len(HOTEL_DATASET)), format_func=lambda x: f"Sample {x+1}")
            
            selected_sample = HOTEL_DATASET[sample_idx]
            sample_text = selected_sample['text']
            ground_truth = selected_sample['entities']
            
            # Use cached predictions instead of re-running the model
            cached_predictions = results['all_predictions'][sample_idx]
            predicted_filtered = [
                entity for entity in cached_predictions 
                if entity.get('score', 1.0) >= cached_threshold
            ]
            
            st.write("**Original Text:**")
            highlighted_text = highlight_entities(sample_text, predicted_filtered)
            st.markdown(highlighted_text, unsafe_allow_html=True)
            
            # Show ground truth vs predictions
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Ground Truth Entities:**")
                gt_data = []
                for entity in ground_truth:
                    gt_data.append({
                        "Entity": entity["text"],
                        "Label": entity["label"],
                        "Position": f"{entity['start']}-{entity['end']}"
                    })
                st.dataframe(gt_data, use_container_width=True)
            
            with col2:
                st.write("**Predicted Entities:**")
                pred_data = []
                for entity in predicted_filtered:
                    pred_data.append({
                        "Entity": entity["text"],
                        "Label": entity["label"],
                        "Confidence": f"{entity.get('score', 1.0):.3f}",
                        "Position": f"{entity['start']}-{entity['end']}"
                    })
                st.dataframe(pred_data, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Model:** [urchade/gliner_multi-v2.1](https://huggingface.co/urchade/gliner_multi-v2.1) | "
        "**Framework:** [GLiNER](https://github.com/urchade/GLiNER) | "
        "**Built with:** [Streamlit](https://streamlit.io/)"
    )

if __name__ == "__main__":
    main()