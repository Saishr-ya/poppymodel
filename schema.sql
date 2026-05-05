-- db/schema.sql
-- Drug Repurposing Engine — PostgreSQL Schema
-- Run with: psql -d repurposing_db -f db/schema.sql

-- ──────────────────────────────────────────────────────────────────────────────
-- DRUGS
-- Primary source: ChEMBL + DrugBank
-- ──────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS drugs (
    chembl_id           VARCHAR(20) PRIMARY KEY,
    name                VARCHAR(255) NOT NULL,
    synonyms            JSONB,                          -- trade names, alternative names
    molecule_type       VARCHAR(50),                    -- 'Small molecule', 'Biologic', etc.
    chirality           VARCHAR(50),                    -- '0'=Racemic, '1'=Single stereoisomer, '2'=Achiral
    oral                BOOLEAN,
    oral_bioavailability_pct FLOAT,
    half_life_hours     FLOAT,
    mw                  FLOAT,
    logp                FLOAT,
    hbd                 INTEGER,                        -- H-bond donors
    hba                 INTEGER,                        -- H-bond acceptors
    bcs_class           VARCHAR(5),                     -- I, II, III, IV
    patent_expiry_year  INTEGER,
    first_approval_year INTEGER,
    black_box_warning   BOOLEAN DEFAULT FALSE,
    withdrawn_flag      BOOLEAN DEFAULT FALSE,
    cyp_substrates      JSONB,                          -- ['CYP2C19', 'CYP3A4', ...]
    cyp_inhibitors      JSONB,
    herg_ic50_um        FLOAT,
    raw_chembl          JSONB,
    raw_drugbank        JSONB,
    updated_at          TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_drugs_chirality ON drugs(chirality);
CREATE INDEX IF NOT EXISTS idx_drugs_oral ON drugs(oral);
CREATE INDEX IF NOT EXISTS idx_drugs_approval ON drugs(first_approval_year);


-- ──────────────────────────────────────────────────────────────────────────────
-- DISEASES
-- Primary source: Orphanet + OMIM
-- ──────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS diseases (
    disease_id          VARCHAR(30) PRIMARY KEY,        -- 'ORPHA:12345' or 'OMIM:123456'
    name                VARCHAR(255) NOT NULL,
    id_source           VARCHAR(20),                    -- 'ORPHANET' or 'OMIM'
    orphanet_id         VARCHAR(20),
    omim_id             VARCHAR(20),
    causal_genes        JSONB,                          -- [{gene_symbol, uniprot_id, score, assoc_type}]
    age_of_onset        VARCHAR(50),                    -- 'Neonatal', 'Childhood', 'Adult', etc.
    prevalence_global_per_million FLOAT,
    prevalence_india_estimate INT,                      -- estimated Indian patient count
    primary_affected_tissue VARCHAR(100),
    disease_subtype_ids JSONB,                          -- [orpha_id, ...]
    natural_history_level INTEGER,                      -- 1–5 scale (5=well-documented)
    hpo_terms           JSONB,                          -- [{hpo_id, name}]
    is_metabolic        BOOLEAN DEFAULT FALSE,          -- inborn error of metabolism
    has_indian_founder_variant BOOLEAN DEFAULT FALSE,
    updated_at          TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_diseases_onset ON diseases(age_of_onset);
CREATE INDEX IF NOT EXISTS idx_diseases_metabolic ON diseases(is_metabolic);


-- ──────────────────────────────────────────────────────────────────────────────
-- CANDIDATE PAIRS
-- One row per (drug, disease) combination scored by the engine
-- ──────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS candidate_pairs (
    id                  SERIAL PRIMARY KEY,
    drug_id             VARCHAR(20) REFERENCES drugs(chembl_id),
    disease_id          VARCHAR(30) REFERENCES diseases(disease_id),

    -- Layer 1A: Target Overlap
    score_target_overlap_jaccard FLOAT,
    score_pathway_enrichment_pvalue FLOAT,

    -- Layer 1B: Network Proximity
    score_network_proximity FLOAT,                      -- avg shortest path hops; lower = better

    -- Layer 2: Transcriptomics (populated when layer is built)
    score_transcriptomic_ks FLOAT,

    -- Layer 3: KG Embedding (populated when layer is built)
    score_kg_cosine     FLOAT,

    -- Layer 4: ADMET
    score_admet_composite FLOAT,
    score_bcs_class     VARCHAR(5),
    score_herg_ic50_um  FLOAT,

    -- Layer 5: Literature
    score_pubmed_cooccurrence FLOAT,
    score_clinical_trial_evidence INTEGER,              -- 0–5
    score_case_report_count INTEGER,

    -- Layer 6: Business subscores (1–5 each)
    score_ip            INTEGER,
    score_regulatory    INTEGER,
    score_market        INTEGER,
    score_manufacturing INTEGER,
    score_clinical_adoption INTEGER,
    score_speed         INTEGER,
    score_business_total INTEGER,                       -- /30; only pursue ≥ 24

    -- PGx
    score_pgx_risk      FLOAT,                         -- 0–1
    pgx_cyp_substrates  JSONB,

    -- Composite
    composite_score     FLOAT,
    rank                INTEGER,

    -- Flags
    flag_disqualified   BOOLEAN DEFAULT FALSE,
    flag_disqualify_reason VARCHAR(255),
    flag_patent_conflict BOOLEAN DEFAULT FALSE,
    flag_herg           BOOLEAN DEFAULT FALSE,
    flag_faers          BOOLEAN DEFAULT FALSE,
    flag_bioavailability BOOLEAN DEFAULT FALSE,
    flag_lipinski_violations INTEGER DEFAULT 0,
    flag_ddi_risk       BOOLEAN DEFAULT FALSE,
    flag_pgx_high_risk  BOOLEAN DEFAULT FALSE,
    flag_polymorph_risk BOOLEAN DEFAULT FALSE,
    flag_pediatric_formulation BOOLEAN DEFAULT FALSE,
    flag_founder_variant BOOLEAN DEFAULT FALSE,

    -- Traceability
    data_sources        JSONB,                          -- {layer_name: {version, timestamp, status}}
    engine_version      VARCHAR(20),
    run_id              VARCHAR(50),                    -- UUID for each batch run
    created_at          TIMESTAMP DEFAULT NOW(),
    updated_at          TIMESTAMP DEFAULT NOW(),

    UNIQUE(drug_id, disease_id, run_id)
);

CREATE INDEX IF NOT EXISTS idx_pairs_composite ON candidate_pairs(composite_score DESC);
CREATE INDEX IF NOT EXISTS idx_pairs_business ON candidate_pairs(score_business_total DESC);
CREATE INDEX IF NOT EXISTS idx_pairs_disqualified ON candidate_pairs(flag_disqualified);
CREATE INDEX IF NOT EXISTS idx_pairs_run ON candidate_pairs(run_id);


-- ──────────────────────────────────────────────────────────────────────────────
-- GROUND TRUTH
-- Known positive and negative drug-disease pairs for validation
-- ──────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS ground_truth (
    drug_id             VARCHAR(20),
    disease_id          VARCHAR(30),
    label               INTEGER NOT NULL,               -- 1 = known positive, 0 = known negative
    evidence_source     VARCHAR(255),                   -- e.g., 'DrugBank approved', 'Phase III failure NCT01234'
    notes               TEXT,
    added_by            VARCHAR(100),
    added_at            TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (drug_id, disease_id)
);


-- ──────────────────────────────────────────────────────────────────────────────
-- VALIDATION RUNS
-- Track validation metrics over time as the engine improves
-- ──────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS validation_runs (
    id                  SERIAL PRIMARY KEY,
    run_id              VARCHAR(50) NOT NULL,
    engine_version      VARCHAR(20),
    auroc               FLOAT,
    precision_at_20     FLOAT,
    n_pairs_scored      INTEGER,
    n_disqualified      INTEGER,
    n_gt_positives_matched INTEGER,
    n_gt_negatives_matched INTEGER,
    false_negative_analysis JSONB,
    notes               TEXT,
    run_at              TIMESTAMP DEFAULT NOW()
);


-- ──────────────────────────────────────────────────────────────────────────────
-- COMPETITOR MONITORING
-- Tracks new patent filings and trial registrations for active candidates
-- ──────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS competitor_alerts (
    id                  SERIAL PRIMARY KEY,
    drug_id             VARCHAR(20),
    disease_id          VARCHAR(30),
    alert_type          VARCHAR(50),                    -- 'PATENT', 'TRIAL_REGISTRATION', 'PUBLICATION'
    alert_source        VARCHAR(100),                   -- 'USPTO', 'ClinicalTrials.gov', etc.
    external_id         VARCHAR(100),                   -- Patent number or NCT ID
    title               TEXT,
    detected_at         TIMESTAMP DEFAULT NOW(),
    reviewed            BOOLEAN DEFAULT FALSE,
    review_notes        TEXT
);

CREATE INDEX IF NOT EXISTS idx_alerts_reviewed ON competitor_alerts(reviewed, detected_at);
