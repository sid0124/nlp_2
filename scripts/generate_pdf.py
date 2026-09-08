"""Generate comprehensive technical PDF report for ARIS project.
Covers Introduction, Models, Algorithms, and Feature Engineering.
Formatted cleanly across 3 structured pages.
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    PageBreak,
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and render total page numbers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        
        # Header (pages 2+)
        if self._pageNumber > 1:
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(colors.HexColor("#1E3A8A"))
            self.drawString(40, 756, "ARIS")
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#64748B"))
            self.drawString(68, 756, "|  Technical Report: Introduction, Models, Algorithms & Features")
            self.drawRightString(572, 756, "Academic Research Intelligence System")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.75)
            self.line(40, 748, 572, 748)

        # Footer (all pages)
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(40, 30, "ARIS — Academic Research Paper Classification & Intelligence")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(572, 30, page_str)
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.75)
        self.line(40, 40, 572, 40)
        
        self.restoreState()


def build_pdf(filename: str):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=48,
        bottomMargin=48,
    )

    styles = getSampleStyleSheet()

    # Custom color palette
    primary_color = colors.HexColor("#1E3A8A")     # Deep Navy
    secondary_color = colors.HexColor("#2563EB")   # Royal Blue
    dark_slate = colors.HexColor("#1E293B")        # Text Dark Slate
    card_bg = colors.HexColor("#F8FAFC")           # Off-white card fill
    border_color = colors.HexColor("#E2E8F0")

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=primary_color,
        spaceAfter=2,
    )

    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#475569"),
        spaceAfter=6,
    )

    meta_style = ParagraphStyle(
        "DocMeta",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10.5,
        textColor=secondary_color,
    )

    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=14.5,
        textColor=primary_color,
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=11.5,
        textColor=secondary_color,
        spaceBefore=6,
        spaceAfter=2,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.2,
        leading=11,
        textColor=dark_slate,
        spaceAfter=4,
    )

    bullet_style = ParagraphStyle(
        "BulletText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.2,
        leading=11,
        textColor=dark_slate,
        leftIndent=10,
        spaceAfter=3,
    )

    formula_style = ParagraphStyle(
        "FormulaBox",
        parent=styles["Normal"],
        fontName="Courier-Oblique",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1E3A8A"),
        alignment=1, # Center
        spaceBefore=2,
        spaceAfter=2,
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.8,
        leading=10,
        textColor=colors.white,
    )

    table_body_style = ParagraphStyle(
        "TableBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=dark_slate,
    )

    table_body_bold = ParagraphStyle(
        "TableBodyBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9.5,
        textColor=primary_color,
    )

    story = []

    # =========================================================================
    # PAGE 1: TITLE, INTRODUCTION, AND BASELINE MODELS
    # =========================================================================
    story.append(Paragraph("Academic Research Intelligence System (ARIS)", title_style))
    story.append(Paragraph("Technical Report: Introduction, Models, Algorithms & Feature Engineering", subtitle_style))
    
    # Metadata Pill Box
    meta_data = [
        [
            Paragraph("<b>Stack:</b> PyTorch 2.14, Transformers, Scikit-Learn, FastAPI", meta_style),
            Paragraph("<b>Taxonomy:</b> 11 Computer Science Subfields / 26 Broad Fields", meta_style),
            Paragraph("<b>Corpus:</b> OpenAlex (CC0) Multi-Domain", meta_style),
        ]
    ]
    meta_table = Table(meta_data, colWidths=[200, 190, 142])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#EFF6FF")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#BFDBFE")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 4))

    # --- SECTION 1: INTRODUCTION ---
    story.append(Paragraph("1. Introduction", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.75, color=primary_color, spaceAfter=4, spaceBefore=0))
    
    intro_p1 = (
        "<b>Project Background & Motivation:</b> The exponential growth of scientific literature presents a formidable challenge "
        "for knowledge discovery, academic literature cataloging, and research governance. Hundreds of thousands of preprints "
        "and peer-reviewed articles are published annually. Traditional classification pipelines either treat papers as flat "
        "bag-of-words (ignoring semantic hierarchies) or feed truncated text into standard Transformer models (e.g., BERT) that "
        "enforce an arbitrary 512-token limit (due to quadratic self-attention complexity). Consequently, over 90% of a full paper's content "
        "is discarded, and section-level context (e.g., distinguishing novel methodology from background citations) is permanently lost."
    )
    story.append(Paragraph(intro_p1, body_style))

    intro_p2 = (
        "<b>Core Objectives:</b> The Academic Research Intelligence System (ARIS) solves this via an end-to-end, explainable "
        "classification framework. ARIS encodes papers using domain-specific scientific embeddings (SciBERT) coupled with a "
        "six-level <b>Hierarchical Attention Network (HAN)</b>. Instead of black-box classification, the system extracts and exposes "
        "the network's <i>real, differentiable attention weights</i> across both sentences and sections, visualising the exact evidence "
        "the model utilized for its prediction."
    )
    story.append(Paragraph(intro_p2, body_style))

    intro_p3 = (
        "<b>Dataset & Label Taxonomy:</b> ARIS is trained on real-world papers ingested from the <b>OpenAlex API</b> (CC0 open license). "
        "The primary benchmark spans the <b>11 Computer Science subfields</b> (Artificial Intelligence, Computer Vision, Signal Processing, "
        "Hardware, Software, Networks, HCI, etc.) — classes deliberately selected for their high terminological overlap and confusable boundaries. "
        "A broader 26-domain field-level taxonomy across physics, biology, and medicine is also supported via configuration."
    )
    story.append(Paragraph(intro_p3, body_style))
    story.append(Spacer(1, 4))

    # --- SECTION 2: MODEL ARCHITECTURE (PART 1: BASELINES) ---
    story.append(Paragraph("2. Model Architecture & Baselines", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.75, color=primary_color, spaceAfter=4, spaceBefore=0))

    model_overview = (
        "All models in the system adhere strictly to a unified <code>BaseClassifier</code> contract (<code>fit</code>, "
        "<code>predict</code>, <code>predict_proba</code>, <code>classes_</code>), ensuring reproducible, apples-to-apples evaluation."
    )
    story.append(Paragraph(model_overview, body_style))

    story.append(Paragraph("2.1 Classical Baselines (Milestone 1)", h2_style))
    b1_text = (
        "• <b>TF-IDF + Logistic Regression:</b> Establishes the classical bag-of-words floor. Uses sublinear term frequency, "
        "word n-grams (1, 2), class-balanced weighting, and the <b>L-BFGS</b> solver for multinomial cross-entropy optimization.<br/>"
        "• <b>TF-IDF + Linear Support Vector Machine (LinearSVC):</b> Computes max-margin hyperplane separation. Since raw SVM "
        "margins are signed uncalibrated distances ($w^T x + b$), ARIS wraps LinearSVC in <b>Platt Scaling</b> "
        "(<code>CalibratedClassifierCV</code> via 5-fold cross-validation on train split) to yield genuine posterior probabilities in [0, 1]."
    )
    story.append(Paragraph(b1_text, bullet_style))

    story.append(Paragraph("2.2 Transformer Baselines (Milestone 2)", h2_style))
    b2_text = (
        "• <b>SciBERT + Linear Head:</b> Evaluates raw scientific contextual embeddings without recurrence. Uses "
        "<code>allenai/scibert_scivocab_uncased</code> (pretrained on 1.14M Semantic Scholar papers). Sentence representations "
        "are extracted via mean pooling and fed into a linear classification head."
    )
    story.append(Paragraph(b2_text, bullet_style))

    # PAGE BREAK TO PAGE 2
    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: FLAGSHIP HAN ARCHITECTURE & ALGORITHMS
    # =========================================================================
    story.append(Paragraph("2.3 Flagship Model: SciBERT + Hierarchical Attention Network (HAN)", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.75, color=primary_color, spaceAfter=4, spaceBefore=0))

    han_desc = (
        "Long research papers are decomposed into a structured hierarchy: <i>Document → Sections → Sentences</i>. "
        "The HAN processes this hierarchy across six modular levels, capturing both intra-section and inter-section semantics:"
    )
    story.append(Paragraph(han_desc, body_style))

    han_levels = [
        "<b>Level 0 (SciBERT Sentence Encoding):</b> Sentences are converted to 768-dimensional dense vectors x_{i,j} via frozen SciBERT with token mean-pooling. Vectors are disk-cached by SHA-256 hash to eliminate redundant CPU computation.",
        "<b>Level 1 (Sentence BiGRU Encoder):</b> A Bidirectional GRU reads sentence vectors sequentially within section i, producing forward and backward hidden states: h_{i,j} = [GRU_fwd(x_{i,j}) ; GRU_bwd(x_{i,j})] in R^{256}.",
        "<b>Level 2 (Sentence Additive Attention):</b> A trainable additive context projection computes normalized attention weights α_{i,j} = softmax(v_sent^T tanh(W_sent · h_{i,j})). The section representation is formed as s_i = Σ_j α_{i,j} h_{i,j}.",
        "<b>Level 3 (Section BiGRU Encoder):</b> A second BiGRU captures document-level narrative flow across the sequence of section vectors: h_i = [GRU_fwd(s_i) ; GRU_bwd(s_i)] in R^{256}.",
        "<b>Level 4 (Section Additive Attention):</b> Evaluates the relative significance of each section: β_i = softmax(v_sec^T tanh(W_sec · h_i)). The comprehensive document vector is d = Σ_i β_i h_i.",
        "<b>Levels 5 & 6 (Dropout & Classifier):</b> Applies dropout (p=0.5) for regularization, followed by a linear classification projection and softmax activation: y_pred = softmax(W_clf · d + b_clf).",
    ]
    for lvl in han_levels:
        story.append(Paragraph(f"• {lvl}", bullet_style))

    story.append(Spacer(1, 2))
    story.append(Paragraph("<i>Bahdanau Additive Attention Formulation:</i>", meta_style))
    story.append(Paragraph("u_{i,j} = tanh(W_s · h_{i,j}),   α_{i,j} = exp(u_{i,j}^T · v_s) / Σ_k exp(u_{i,k}^T · v_s),   s_i = Σ_j α_{i,j} · h_{i,j}", formula_style))
    story.append(Paragraph(
        "<b>Real Explainable Attention:</b> Unlike post-hoc perturbations (e.g., LIME or SHAP), ARIS captures the exact internal "
        "differentiable attention weights during inference. These are aligned back to text spans and served directly to the dashboard.",
        body_style
    ))
    story.append(Spacer(1, 4))

    # --- SECTION 3: ALGORITHMS ---
    story.append(Paragraph("3. Algorithms & Computational Methods", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.75, color=primary_color, spaceAfter=4, spaceBefore=0))

    algo_p1 = (
        "ARIS incorporates specialized algorithms across every stage of the lifecycle, from raw data hygiene to neural optimization:"
    )
    story.append(Paragraph(algo_p1, body_style))

    algos_data = [
        [
            Paragraph("Algorithm / Method", table_header_style),
            Paragraph("Module / File", table_header_style),
            Paragraph("Algorithmic Details & Complexity", table_header_style),
        ],
        [
            Paragraph("<b>Bottom-k MinHash Shingle Sketch</b>", table_body_bold),
            Paragraph("<code>src/data_pipeline/dedup.py</code>", table_body_style),
            Paragraph("Avoids O(N^2) pairwise checks across 16k papers (135M pairs). Hashes 5-word shingles; retains only the k=64 smallest hashes in an inverted bucket index. Computes exact Jaccard similarity (s >= 0.85) only within buckets, achieving near-linear O(N log N) scalability.", table_body_style),
        ],
        [
            Paragraph("<b>Stratified Split & Leakage Audit</b>", table_body_bold),
            Paragraph("<code>src/data_pipeline/split.py</code>", table_body_style),
            Paragraph("Enforces strict 70/15/15 train/val/test partitioning preserving class ratios. Post-split audit verifies with SHA-256 and shingle checks that no identical or near-duplicate paper spans across split boundaries.", table_body_style),
        ],
        [
            Paragraph("<b>Academic Sentence Boundary Disambiguation</b>", table_body_bold),
            Paragraph("<code>src/preprocessing/sentences.py</code>", table_body_style),
            Paragraph("Rule-based tokenizer with backward lookahead. Prevents false splits on academic abbreviations (<i>et al., i.e., e.g., fig., eq.</i>), author initials (<i>J. Smith</i>), decimals (<i>0.05</i>), and versioned headers (<i>Sec. 2.1</i>). Outputs exact character spans.", table_body_style),
        ],
        [
            Paragraph("<b>Canonical Section Mapping</b>", table_body_bold),
            Paragraph("<code>src/preprocessing/sections.py</code>", table_body_style),
            Paragraph("Regex pattern-matching engine mapping diverse paper headings into 11 canonical types (<i>Abstract, Introduction, Related Work, Methodology, Experiments, Results, Discussion, Conclusion, etc.</i>).", table_body_style),
        ],
        [
            Paragraph("<b>AdamW with Gradient Clipping</b>", table_body_bold),
            Paragraph("<code>src/models/han_classifier.py</code>", table_body_style),
            Paragraph("Optimizes HAN parameters using AdamW (learning rate 2e-5, weight decay 0.01). Applies gradient norm clipping at 1.0 to prevent exploding gradients in recurrent BiGRU layers.", table_body_style),
        ],
        [
            Paragraph("<b>Class Balancing & Focal Loss</b>", table_body_bold),
            Paragraph("<code>src/models/han_classifier.py</code>", table_body_style),
            Paragraph("Dynamic inverse class-frequency loss weighting (w_c = N / (C · N_c)) combined with optional Focal Loss (γ = 2.0) to counteract majority-class bias without artificial text resampling.", table_body_style),
        ],
        [
            Paragraph("<b>Platt Probability Calibration</b>", table_body_bold),
            Paragraph("<code>src/models/baselines.py</code>", table_body_style),
            Paragraph("Fits a logistic sigmoid function over LinearSVC decision margins via 5-fold cross-validation on train data, transforming arbitrary distances into true posterior probabilities.", table_body_style),
        ],
        [
            Paragraph("<b>Vector Cosine Similarity Retrieval</b>", table_body_bold),
            Paragraph("<code>src/api/retrieval.py</code>", table_body_style),
            Paragraph("Computes pairwise cosine similarities across normalized paper embeddings to provide instantaneous k-NN similar paper recommendations on the dashboard.", table_body_style),
        ],
    ]

    algo_table = Table(algos_data, colWidths=[130, 115, 287])
    algo_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, card_bg]),
    ]))
    story.append(algo_table)

    # PAGE BREAK TO PAGE 3
    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: FEATURE ENGINEERING & SUMMARY COMPARISON TABLE
    # =========================================================================
    story.append(Paragraph("4. Feature Engineering Steps", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.75, color=primary_color, spaceAfter=4, spaceBefore=0))

    feat_intro = (
        "ARIS utilizes a multi-tiered feature engineering pipeline transforming raw publisher records and unstructured text "
        "into dense, hierarchically structured representations:"
    )
    story.append(Paragraph(feat_intro, body_style))

    steps_data = [
        ("Step 1: Inverted Index Abstract Reconstruction",
         "OpenAlex supplies abstracts as inverted indices (e.g., <code>{'deep': [0], 'learning': [1]}</code>) to compress payload sizes. "
         "The pipeline reconstructs the original linear text sequence by sorting token occurrences by position index."),
        
        ("Step 2: Dual-Tier Text Normalization Strategy",
         "• <b>Conservative Model Normalization (<code>clean_text</code>):</b> Prepares clean text for model consumption. Applies Unicode NFKC "
         "normalization; decodes HTML entities (<code>&amp;</code> → <code>&</code>); strips JATS markup; "
         "replaces URLs/DOIs with spaces; collapses whitespace. <i>Crucially preserves casing and punctuation</i> to retain vital acronym semantics (e.g., 'BERT', 'GAN', 'US' vs 'us').<br/>"
         "• <b>Aggressive Matching Normalization (<code>normalize_for_matching</code>):</b> Used exclusively for deduplication and hashing. Enforces Unicode NFKC, "
         "full case-folding (<code>casefold()</code>), and strips all punctuation to ensure cosmetic variants hash identically."),
        
        ("Step 3: Section Extraction & Canonical Alignment",
         "Decomposes full documents into logical sections based on double-newline breaks and header heuristics. Section names are canonically mapped "
         "(e.g., '3. Experimental Setup' → <code>experiments</code>), enabling section-aware attention scoring."),
        
        ("Step 4: Hierarchical Long-Document Bounding",
         "To handle documents of arbitrary length without out-of-memory errors or arbitrary truncation: limits documents to <b>max 8 sections</b> "
         "and <b>max 32 sentences per section</b>. This yields a uniform 4-D tensor batch layout <code>(batch, max_sec=8, max_sent=32, dim=768)</code> "
         "paired with boolean masks to handle variable-length documents."),
        
        ("Step 5A: Classical Feature Vectorization (TF-IDF)",
         "Extracts up to 200,000 unigram and bigram features (<code>ngram_range=[1, 2]</code>) capturing compound technical terminology. "
         "Employs <b>sublinear term frequency</b> (1 + log(tf)) to dampen repetitive term dominance. Filters extreme frequencies "
         "with min_df=2 (eliminating typos) and max_df=0.90 (removing corpus-wide stop words)."),
        
        ("Step 5B: Contextual Embeddings with On-Disk Caching",
         "Passes sentences through <code>allenai/scibert_scivocab_uncased</code> with a token sequence bound of 256. Applies <b>mean pooling</b> "
         "over non-padded token vectors to construct 768-dimensional sentence vectors. Caches embeddings on disk keyed by SHA-256 hash of "
         "<code>(model_name, pooling_mode, sentence_text)</code>, ensuring zero redundant computations across inference runs."),
    ]

    for title, desc in steps_data:
        story.append(Paragraph(f"<b>{title}</b>", h2_style))
        story.append(Paragraph(desc, body_style))

    story.append(Spacer(1, 3))

    # --- SECTION 5: SUMMARY COMPARISON TABLE ---
    story.append(Paragraph("5. Summary Comparison: Classical vs. Neural Stack", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.75, color=primary_color, spaceAfter=4, spaceBefore=0))

    comp_data = [
        [
            Paragraph("Technical Dimension", table_header_style),
            Paragraph("Classical Stack (Baselines 1 & 2)", table_header_style),
            Paragraph("Neural Hierarchical Stack (SciBERT + HAN)", table_header_style),
        ],
        [
            Paragraph("<b>Text Granularity</b>", table_body_bold),
            Paragraph("Flat document (Title + Abstract concatenated)", table_body_style),
            Paragraph("Hierarchical: Document → Sections → Sentences", table_body_style),
        ],
        [
            Paragraph("<b>Feature Representation</b>", table_body_bold),
            Paragraph("200,000-dim Sublinear TF-IDF (Unigrams + Bigrams)", table_body_style),
            Paragraph("768-dim SciBERT dense vectors per sentence", table_body_style),
        ],
        [
            Paragraph("<b>Sequence & Context Modeling</b>", table_body_bold),
            Paragraph("None (Bag-of-Words independence assumption)", table_body_style),
            Paragraph("Two-level Bidirectional GRUs (Sentence & Section order)", table_body_style),
        ],
        [
            Paragraph("<b>Attention / Explainability</b>", table_body_bold),
            Paragraph("Global feature coefficients (logistic/SVM weights)", table_body_style),
            Paragraph("Real additive attention weights aligned to sentence/section spans", table_body_style),
        ],
        [
            Paragraph("<b>Long Document Processing</b>", table_body_bold),
            Paragraph("Truncated abstract only", table_body_style),
            Paragraph("Structured bounding: 8 sections × 32 sentences tensor", table_body_style),
        ],
        [
            Paragraph("<b>Imbalance Mitigation</b>", table_body_bold),
            Paragraph("Class-balanced weighting in loss function", table_body_style),
            Paragraph("Inverse class-frequency loss + Focal Loss (γ = 2.0)", table_body_style),
        ],
        [
            Paragraph("<b>Confidence Calibration</b>", table_body_bold),
            Paragraph("Platt Scaling (Sigmoid CV) for Linear SVM", table_body_style),
            Paragraph("Native Softmax posterior probability distribution", table_body_style),
        ],
        [
            Paragraph("<b>Hardware Target</b>", table_body_bold),
            Paragraph("Lightweight CPU (seconds per train)", table_body_style),
            Paragraph("CPU/GPU hybrid with on-disk embedding cache", table_body_style),
        ],
    ]

    comp_table = Table(comp_data, colWidths=[120, 195, 217])
    comp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, card_bg]),
    ]))
    story.append(comp_table)

    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated technical report at: {filename}")


if __name__ == "__main__":
    output_path = Path("ARIS_Technical_Report.pdf").resolve()
    build_pdf(str(output_path))

