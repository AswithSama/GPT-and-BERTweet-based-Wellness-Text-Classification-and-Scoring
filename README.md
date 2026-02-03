# LLM Assisted Reflection to Wellness-NLP-Pipeline-Weak-Labeling-with-GPT-and-BERTweet-Fine-Tuning
I led and architected the below project end-to-end as a reflection-based NLP system for Uplifty’s Human Growth Index, designing the methodology, building the dataset labeling pipeline, implementing preprocessing and model training workflows, and coordinating a 5-member team to support large-scale data generation, experimentation, and model evaluation.

# 📖 Introduction

This project originates from **Uplifty**, a platform built with a simple mission: help people improve their everyday lives through consistent habits, meaningful goals, real-world experiences, and authentic human connection.

Rather than encouraging passive scrolling or superficial online engagement, Uplifty promotes **action-driven growth** — following routines, completing challenges, participating in events, and supporting others. To measure this progress holistically, Uplifty introduced the **Human Growth Index (HGI)**, a score that reflects how consistently users take constructive steps toward personal, social, and emotional well-being.

While many behaviors can be tracked directly — such as goal completion or event participation — not all aspects of growth leave clear digital footprints. Important dimensions like mental health, spirituality, learning, compassion, or life transitions often appear only in personal thoughts and experiences.

This insight led to **Reflections** — short daily journal entries where users express their day in their own words. These reflections capture the human side of growth that traditional activity tracking cannot measure.

---

# 🧠 Abstract

This project builds an **end-to-end AI system** that converts users’ daily journal-style reflections into **measurable wellness insights**.

Using **GPT-based weak labeling** to automatically generate training data and a **BERTweet transformer model fine-tuned with PyTorch**, the pipeline processes informal, emotion-rich text — including emojis and conversational language — and performs **multi-label classification** across multiple wellness dimensions such as mental, physical, financial, and social health.

The model then aggregates predictions using a **weighted scoring mechanism** to compute an overall **Human Growth Index**, transforming unstructured personal narratives into clear, quantitative metrics that can be tracked over time.

In short, the system converts:

**free-text reflections → structured wellness signals → actionable well-being metrics**

To demonstrate the solution in a practical and user-friendly way, a **lightweight web application** was also developed to showcase the functionality through an interactive interface.


# ⚙️ Technical Methodology

## 1️⃣ Data Acquisition & Dataset Construction
### 1.1 Problem — No Labeled Reflection Dataset
  To train a supervised multi-label wellness classifier, we require text paired with wellness categories. However, no public dataset directly maps diary-style reflections to structured wellness dimensions such as mental, physical, spiritual, financial, or social health. This created a fundamental bottleneck: we had raw text but no reliable labels for training or evaluation.
  
### 1.2 Automated Label Generation using LLMs
So inorder to overcome the bottleneck one idea is  to do Manual annotation for 100,000+ entries which was infeasible. Instead, we used  GPT APIs to automatically label each record.
Each text was passed through a strict JSON-schema prompt to assign:
| Dimension                          | Label Range |
| ---------------------------------- | ----------- |
| Physical wellness                  | -1 / 0 / 1  |
| Intellectual wellness              | 0 / 1       |
| Occupational wellness              | -1 / 0 / 1  |
| Financial wellness                 | -1 / 0 / 1  |
| Social interaction wellness        | -1 / 0 / 1  |
| Spiritual wellness                 | 0 / 1       |
| Mental wellness                    | -1 / 0 / 1  |
| Compassion / contribution wellness | 0 / 1       |
| Family & caregiving                | -1 / 0 / 1  |
| Leisure & travel                   | 0 / 1       |
| Life events & transitions          | 0 / 1       |
| Neutral                            | 0 / 1       |




### 1.3 Base Text Source Selection
After exploring multiple options (Reddit posts, online diary blogs, scraped personal reflections), we selected Kaggle’s HappyDB dataset, which contains over 100,000 short, clean, diary-style text entries describing everyday experiences and emotions.
Reasons for choosing HappyDB:
- Large scale (100k+ records)
- High quality and cleaned text
- Natural first-person reflections
- Covers diverse life experiences aligned with wellness dimensions
This dataset provided strong semantic coverage for all 11 growth dimensions while being easier to standardize compared to noisy scraped sources.
Approximately 50,000 labeled samples were generated in one day, forming the initial training corpus.

### 1.4 Label Reliability Improvements (Planned)
Since LLM-generated labels can be inconsistent, we planned:
- Confidence filtering
- Revalidation using stronger models
- Prompt optimization
- Strict schema enforcement
These steps improve label stability and reduce noise.
 
## 2. Data Cleaning & Processing
### 2.1 Custom Stopword Strategy
We customized standard stopword removal to preserve context that is critical in reflection-style text. Instead of blindly removing common words, we retained linguistic cues that indicate speaker perspective, negation, and situational meaning, since these directly affect wellness interpretation.
Key adjustments include:
- Preserve negations: not, no, never
- Preserve pronouns: I, me, my, we, they
- Preserve context markers: this, that, here, when, how
- Maintain first-person focus for accurate “user-state” detection
- Avoid loss of semantic polarity caused by removing negation terms
This ensures correct differentiation between personal experiences (e.g., “I am not feeling well”) and external references (e.g., “My friend is not feeling well”), improving classification reliability

### 2.2 Emoji Processing

Reflections often contain emojis that carry strong emotional and contextual meaning (e.g., happiness, stress, frustration, love). Instead of treating emojis as noise or removing them during cleaning, we treat them as semantic features that directly contribute to model predictions.

We implemented a custom emoji-aware preprocessing pipeline that detects emojis using Unicode ranges and converts them into textual tokens compatible with the language model.

**Processing steps include:**

- Detect emojis using a dedicated Unicode regex pattern
- Convert each emoji to its text alias using `emoji.demojize()`
- Example: 😂 → face_with_tears_of_joy
- Keep only aliases that exist in the BERTweet vocabulary (avoid `[UNK]` tokens)
- Collapse consecutive repeated emojis into a single token to prevent noise
- Example: 😂😂😂 → face_with_tears_of_joy
- Preserve emojis directly in the token stream alongside words so they are embedded like regular tokens

This approach allows emotional signals such as 😊, 😔, ❤️, or 😂 to be represented explicitly in the embedding space, enabling the model to better capture sentiment, mood, and affective context present in short personal reflections.


*******************************INCLUDE THE CODE CELL PICTURE HERE **********************
### 2.3 Language Model Selection for Text Style
We selected Meta AI’s BERTweet due to its specialization in informal, short-form, and social-language text similar to diary reflections. Compared to standard BERT models, BERTweet better captures conversational tone and emoji semantics.
Reasons for selection:
- Pretrained on social media languag
- Handles slang, abbreviations, and informal grammar
- Native emoji vocabulary support
- Strong performance on short, personal texts
- More aligned with reflection-style inputs than vanilla BERT
This makes it well-suited for modeling real-world user reflections.
 
## 3. Training Procedures Explored
To determine the most practical and effective way to use Meta AI’s BERTweet for wellness reflection classification, we evaluated three progressively advanced training strategies—starting from a simple frozen baseline, moving to selective fine-tuning, and finally exploring parameter-efficient methods—to balance accuracy, stability, and computational cost.
### 3.1 Approach 1 — Frozen Embeddings (Feature Extraction Only)
In the first approach, we used Meta AI’s BERTweet only as a text encoder. We did not train or modify the transformer at all. Instead, the model simply converted each reflection into numerical embeddings, and those embeddings were passed to traditional machine learning models for classification.
This setup was mainly used to create a quick and reliable baseline. Since no deep learning weights were updated, training was very fast and required very little compute. It also made debugging easier. However, because the language model stayed frozen, it could not learn the specific style or vocabulary of wellness reflections, which limited the final accuracy.
Key points:
- Transformer completely frozen
- Use embeddings with Logistic Regression / Random Forest / LightGBM
- Very fast and low cost
- Stable and simple
- Lower accuracy due to no domain learning
 
### 3.2 Approach 2 — Partial Fine-Tuning (Selected Strategy)
In the second approach, we allowed the model to learn a little. Instead of training all 135 million parameters, we only unfroze the top two or three layers of the transformer and kept the rest frozen. This let the model adapt to wellness-related language while still keeping training efficient.
This turned out to be the best balance. The model learned the domain better, improved accuracy, and still trained quickly. It also converged faster, usually within about nine epochs. Because it offered strong performance without high compute costs, we chose this as the final production method.
Key points:
- Only top layers trained (~2M parameters)
- Learns wellness-specific language
- Faster training than full fine-tuning
- Better accuracy than frozen setup
- Selected as the final approach
 
### 3.3 Approach 3 — Parameter-Efficient Fine-Tuning (Exploratory)
In the third approach, we tested more advanced techniques like LoRA and other PEFT methods. These methods add small trainable components (adapters) instead of updating the whole model. The idea is to get good performance while using very little memory and compute.
Although these methods are promising, they were harder to implement and integrate into our pipeline. Because of these practical challenges, we did not fully adopt them yet. We plan to revisit them later as future improvements.
Key points:
- Train only small adapter modules
- Very low memory and compute usage
- More complex to implement
- Kept for future experimentation
 
#### Final Takeaway (Simple View)
You can think of the three approaches like this:
- Frozen → fastest but less accurate
- Partial fine-tuning → best balance of speed and accuracy
- PEFT → most efficient in theory but harder to set up
Because of this trade-off, partial fine-tuning gave the most practical and reliable results for the system.
 
## 3.4 Hyperparameters Used
| Parameter | Value |
|-----------|-----------|
| Epochs | 9 |
| Learning Rate | 2e-5 (small for stability) |
| Batch Size | 16–32 |
| Max Sequence Length | 128–256 |
| Layers Unfrozen | Top 3 |
| Loss | Multi-label BCEWithLogits |

## 4. Model Saving & Exporting
After training, models were packaged for deployment:
Saved Components
1.	pytorch_model.bin – model weights
3.	Tokenizer – ensures consistent preprocessing
4.	config.json – metadata (labels, thresholds, input size)
This guarantees inference-time reproducibility.
 
## 5. Metrics
Evaluation Results
Precision:
Recall:
F1 Score:
***************************************(Values to be added after final evaluation)********************************************
 
## 6. Website Creation & Deployment
### 6.1 Backend Integration
After training, the model was converted from an offline experiment into a deployable backend service. All inference steps were centralized inside a single app.py pipeline (which is used to deploy the model as a website) so that the same preprocessing, tokenization, and prediction logic used during training are reused during deployment. This keeps behavior consistent and prevents training–serving mismatches across development and production.
When the server starts, it loads the saved model artifacts into memory once for efficiency. The backend then works as a simple processing layer: it accepts reflection text, cleans and tokenizes it, runs the model for inference, and returns probability scores for each wellness dimension. Keeping everything in one modular pipeline makes the system easier to maintain, debug, and scale.
For reliable and reproducible deployment, the components are saved as separate standard files:
- pytorch_model.bin → trained model weights
- Tokenizer files (via save_pretrained()) → consistent preprocessing during inference
- config.json → metadata (model info, labels, thresholds, max input length)
This artifact-based structure ensures the backend can reliably reload the exact same model configuration and simplifies versioning and future updates.
 
## 7. Scoring Mechanism
After classification, predicted categories are combined using a weight-based scoring framework to compute the Human Wellness Score which is later used in Human Growth Index calculation.
Approach:
- Weighted sum of wellness dimensions
- Similar to credit scoring or player rating systems
- Currently manually tuned
- Future work: learned or adaptive weights
# EXAMPLE
Considering the following reflection and the probabilities predicted by the model:
“I went for a short run in the morning and completed my work tasks on time, but I felt mentally drained, worried about my finances, skipped social plans, and struggled to focus in 
#### Predicted probabilities

- Physical → 0.75
- Occupational/Productivity → 0.70
- Mental stress (non-physical) → 0.60
- Financial stress (non-physical) → 0.55
- Social withdrawal (non-physical) → 0.50

#### Example weights

- Physical = +0.35
- Occupational = +0.25
- Mental = -0.20
- Financial = -0.10
- Social = -0.100
Wellness Score calculation
Score = (0.75×0.35) + (0.70×0.25) − (0.60×0.20) − (0.55×0.10) − (0.50×0.10)
Score ≈ 0.21
This way, positive behaviors increase the score while non-physical stress factors reduce it, giving a balanced and realistic Human Wellness Score.


