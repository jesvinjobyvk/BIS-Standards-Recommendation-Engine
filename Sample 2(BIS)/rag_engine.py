"""
BIS Standards RAG Engine
Retrieval-Augmented Generation pipeline for BIS SP 21 Building Materials standards.
Uses TF-IDF + keyword boosting for fast, hallucination-free retrieval.
"""

import re
import math
import json
from collections import defaultdict
from typing import List, Dict, Tuple

# ─────────────────────────────────────────────
# BIS SP 21 Knowledge Base — Building Materials
# ─────────────────────────────────────────────
BIS_STANDARDS = [
    {
        "id": "IS_269",
        "code": "IS 269",
        "title": "Ordinary Portland Cement, 33 Grade — Specification",
        "category": "Cement",
        "description": (
            "Specification for ordinary portland cement 33 grade used for general construction "
            "purposes including plastering, masonry, flooring, and low-strength concrete. "
            "Covers chemical composition, physical requirements, fineness, setting time, "
            "soundness, and compressive strength requirements at 3, 7, and 28 days."
        ),
        "keywords": [
            "cement", "ordinary portland cement", "opc", "33 grade", "33-grade",
            "general construction", "plastering", "masonry", "mortar", "binder",
            "hydraulic cement", "clinker", "gypsum", "low strength concrete"
        ],
        "applications": ["plastering", "masonry mortar", "low-strength concrete", "flooring tiles"],
        "test_standards": ["IS 4031", "IS 650"],
    },
    {
        "id": "IS_8112",
        "code": "IS 8112",
        "title": "Ordinary Portland Cement, 43 Grade — Specification",
        "category": "Cement",
        "description": (
            "Specification for ordinary portland cement 43 grade suitable for general civil "
            "engineering construction works, precast concrete, prestressed concrete, and "
            "moderate-strength structural concrete. Specifies minimum 43 MPa compressive strength "
            "at 28 days. Covers fineness, setting time, soundness, and chemical requirements."
        ),
        "keywords": [
            "cement", "ordinary portland cement", "opc", "43 grade", "43-grade",
            "moderate strength", "structural concrete", "precast", "prestressed",
            "civil construction", "concrete mix", "hydraulic binder"
        ],
        "applications": ["structural concrete", "precast elements", "prestressed concrete", "general construction"],
        "test_standards": ["IS 4031", "IS 650"],
    },
    {
        "id": "IS_12269",
        "code": "IS 12269",
        "title": "Ordinary Portland Cement, 53 Grade — Specification",
        "category": "Cement",
        "description": (
            "Specification for high-strength ordinary portland cement 53 grade used for high-rise "
            "buildings, bridges, runways, industrial floors, and high-performance structural "
            "concrete. Achieves minimum 53 MPa compressive strength at 28 days. Suitable for "
            "rapid construction where early strength gain is critical."
        ),
        "keywords": [
            "cement", "ordinary portland cement", "opc", "53 grade", "53-grade",
            "high strength", "high rise", "bridge", "runway", "industrial floor",
            "high performance concrete", "rapid strength", "structural", "rcc"
        ],
        "applications": ["high-rise buildings", "bridges", "runways", "high-strength RCC"],
        "test_standards": ["IS 4031", "IS 650"],
    },
    {
        "id": "IS_1489_1",
        "code": "IS 1489 (Part 1)",
        "title": "Portland Pozzolana Cement, Fly Ash Based — Specification",
        "category": "Cement",
        "description": (
            "Specification for portland pozzolana cement manufactured by intergrinding portland "
            "cement clinker with fly ash. Offers improved durability, reduced heat of hydration, "
            "and resistance to sulphate attack and alkali-silica reaction. Ideal for mass concrete, "
            "marine structures, sewage works, and humid environments. Fly ash content 15–35%."
        ),
        "keywords": [
            "pozzolana cement", "ppc", "fly ash cement", "portland pozzolana",
            "blended cement", "durability", "sulphate resistance", "marine",
            "mass concrete", "heat of hydration", "humid", "alkali silica reaction"
        ],
        "applications": ["marine structures", "mass concrete", "sewage works", "sulphate-rich soil"],
        "test_standards": ["IS 4031", "IS 1727"],
    },
    {
        "id": "IS_1489_2",
        "code": "IS 1489 (Part 2)",
        "title": "Portland Pozzolana Cement, Calcined Clay Based — Specification",
        "category": "Cement",
        "description": (
            "Specification for portland pozzolana cement using calcined clay as pozzolanic material. "
            "Provides similar benefits to fly ash PPC including sulphate resistance and reduced "
            "permeability. Used in water-retaining structures, foundations in aggressive soils, "
            "and general construction where improved durability is needed."
        ),
        "keywords": [
            "pozzolana cement", "ppc", "calcined clay cement", "portland pozzolana",
            "blended cement", "water retaining", "foundation", "aggressive soil",
            "durability", "reduced permeability", "pozzolanic"
        ],
        "applications": ["water-retaining structures", "foundations", "aggressive soil environments"],
        "test_standards": ["IS 4031", "IS 1727"],
    },
    {
        "id": "IS_455",
        "code": "IS 455",
        "title": "Portland Slag Cement — Specification",
        "category": "Cement",
        "description": (
            "Specification for portland slag cement manufactured using ground granulated blast "
            "furnace slag (GGBS). Provides high resistance to sulphate attack, chloride ingress, "
            "and seawater. Low heat of hydration makes it ideal for mass concrete pours, "
            "marine structures, coastal construction, and underground works."
        ),
        "keywords": [
            "slag cement", "psc", "portland slag", "ggbs", "blast furnace slag",
            "sulphate resistant", "marine", "seawater", "coastal", "chloride",
            "mass concrete", "low heat", "underground"
        ],
        "applications": ["marine construction", "coastal works", "mass concrete dams", "underground structures"],
        "test_standards": ["IS 4031"],
    },
    {
        "id": "IS_12600",
        "code": "IS 12600",
        "title": "Low Heat Portland Cement — Specification",
        "category": "Cement",
        "description": (
            "Specification for low heat portland cement with restricted heat of hydration to "
            "minimise thermal cracking in large concrete pours. Used in dams, large foundations, "
            "thick retaining walls, and other mass concrete structures where temperature rise "
            "must be controlled."
        ),
        "keywords": [
            "low heat cement", "mass concrete", "dam", "large foundation",
            "thermal cracking", "heat of hydration", "retaining wall", "thick slab",
            "temperature control", "concrete dam"
        ],
        "applications": ["concrete dams", "mass foundations", "thick retaining walls"],
        "test_standards": ["IS 4031"],
    },
    {
        "id": "IS_8041",
        "code": "IS 8041",
        "title": "Rapid Hardening Portland Cement — Specification",
        "category": "Cement",
        "description": (
            "Specification for rapid hardening portland cement that achieves high early strength "
            "within 24–72 hours. Used for emergency repairs, precast production, cold weather "
            "concreting, pavement repair, and where formwork needs to be stripped quickly."
        ),
        "keywords": [
            "rapid hardening cement", "early strength", "quick setting", "fast set",
            "emergency repair", "precast", "cold weather", "pavement repair",
            "formwork stripping", "rhpc"
        ],
        "applications": ["emergency repairs", "precast production", "cold weather concreting"],
        "test_standards": ["IS 4031"],
    },
    {
        "id": "IS_6452",
        "code": "IS 6452",
        "title": "High Alumina Cement for Structural Use — Specification",
        "category": "Cement",
        "description": (
            "Specification for high alumina cement used in refractory applications, industrial "
            "furnace linings, and high-temperature resistant structures. Also used where rapid "
            "strength gain and resistance to chemical attack from acids and alkalis is required."
        ),
        "keywords": [
            "high alumina cement", "hac", "refractory", "furnace lining", "high temperature",
            "acid resistant", "chemical attack", "rapid strength", "industrial"
        ],
        "applications": ["furnace linings", "refractory structures", "chemical-resistant applications"],
        "test_standards": ["IS 4031"],
    },
    {
        "id": "IS_3466",
        "code": "IS 3466",
        "title": "Masonry Cement — Specification",
        "category": "Cement",
        "description": (
            "Specification for masonry cement used in mortar for brick, block, and stone masonry. "
            "Provides good workability, water retention, and bond strength. Not suitable for "
            "structural concrete. Covers physical and chemical requirements."
        ),
        "keywords": [
            "masonry cement", "brick mortar", "block mortar", "stone masonry",
            "plastering", "brickwork", "workability", "bond strength", "mortar mix"
        ],
        "applications": ["brick masonry", "block masonry", "stone masonry", "plastering mortar"],
        "test_standards": ["IS 4031"],
    },

    # ─── STEEL ───────────────────────────────────────────────────────────────
    {
        "id": "IS_1786",
        "code": "IS 1786",
        "title": "High Strength Deformed Steel Bars and Wires for Concrete Reinforcement — Specification",
        "category": "Steel",
        "description": (
            "Specification for high strength deformed (HSD) steel bars and wires used as concrete "
            "reinforcement. Covers Fe415, Fe415D, Fe500, Fe500D, Fe550, Fe550D, and Fe600 grades. "
            "Specifies mechanical properties, chemical composition, rib geometry, bendability, "
            "and weldability. The primary standard for TMT bars used in RCC construction."
        ),
        "keywords": [
            "tmt bar", "tmt", "deformed bar", "hsd bar", "rebar", "reinforcement bar",
            "steel reinforcement", "fe415", "fe500", "fe500d", "fe550", "fe600",
            "high strength", "ribbed bar", "concrete reinforcement", "rcc", "structural steel"
        ],
        "applications": ["RCC slabs", "beams", "columns", "foundations", "bridges"],
        "test_standards": ["IS 1608", "IS 1599"],
    },
    {
        "id": "IS_432_1",
        "code": "IS 432 (Part 1)",
        "title": "Mild Steel and Medium Tensile Steel Bars and Hard-Drawn Steel Wire for Concrete Reinforcement — Mild Steel and Medium Tensile Steel Bars",
        "category": "Steel",
        "description": (
            "Specification for mild steel (Fe250) and medium tensile steel bars used as plain "
            "round bars in concrete reinforcement. Covers yield strength, tensile strength, "
            "elongation, and bend test requirements. Used in stirrups, ties, and secondary "
            "reinforcement where ductility is more important than high strength."
        ),
        "keywords": [
            "mild steel bar", "plain bar", "round bar", "fe250", "stirrup",
            "ties", "secondary reinforcement", "ductile", "low carbon steel", "ms bar"
        ],
        "applications": ["stirrups", "ties", "secondary reinforcement", "small structures"],
        "test_standards": ["IS 1608"],
    },
    {
        "id": "IS_2062",
        "code": "IS 2062",
        "title": "Hot Rolled Medium and High Tensile Structural Steel — Specification",
        "category": "Steel",
        "description": (
            "Specification for hot rolled structural steel in plates, sections, and bars used in "
            "general structural fabrication. Covers E250, E300, E350, E410, E450, E550 grades. "
            "Used for fabricated steel structures, industrial buildings, trusses, bridges, "
            "crane girders, and general structural purposes."
        ),
        "keywords": [
            "structural steel", "hot rolled steel", "e250", "e350", "steel plate",
            "steel section", "fabrication", "industrial building", "truss", "girder",
            "crane runway", "structural fabrication", "ismc", "ismb", "angle section"
        ],
        "applications": ["steel structures", "industrial buildings", "bridges", "trusses"],
        "test_standards": ["IS 1608", "IS 1599"],
    },
    {
        "id": "IS_808",
        "code": "IS 808",
        "title": "Dimensions for Hot Rolled Steel Beam, Column, Channel and Angle Sections — Specification",
        "category": "Steel",
        "description": (
            "Specification for standard dimensions and sectional properties of hot rolled steel "
            "sections including I-beams (ISMB), column sections (ISHB), channel sections (ISMC), "
            "and angle sections (ISA). Used as reference for structural design and procurement."
        ),
        "keywords": [
            "steel section", "ismb", "ishb", "ismc", "isa", "i beam", "h beam",
            "channel section", "angle section", "steel dimensions", "structural section",
            "rolled section", "beam section", "column section"
        ],
        "applications": ["structural design", "steel fabrication", "procurement specifications"],
        "test_standards": ["IS 2062"],
    },
    {
        "id": "IS_1161",
        "code": "IS 1161",
        "title": "Steel Tubes for Structural Purposes — Specification",
        "category": "Steel",
        "description": (
            "Specification for welded and seamless steel tubes used for structural purposes "
            "including hollow sections in trusses, frames, scaffolding, and structural applications. "
            "Covers dimensions, mechanical properties, and testing requirements."
        ),
        "keywords": [
            "steel tube", "hollow section", "rhs", "shs", "scaffolding tube",
            "structural tube", "circular hollow section", "chs", "steel pipe structural"
        ],
        "applications": ["structural hollow sections", "trusses", "scaffolding", "frames"],
        "test_standards": ["IS 1608"],
    },

    # ─── CONCRETE ────────────────────────────────────────────────────────────
    {
        "id": "IS_456",
        "code": "IS 456",
        "title": "Plain and Reinforced Concrete — Code of Practice",
        "category": "Concrete",
        "description": (
            "Comprehensive code of practice for the structural use of plain and reinforced "
            "concrete. Covers concrete grades M10 to M55, mix design, durability, cover to "
            "reinforcement, workability, water-cement ratio, curing, formwork, and structural "
            "design requirements. The foundational standard for all RCC construction in India."
        ),
        "keywords": [
            "reinforced concrete", "rcc", "concrete grade", "m20", "m25", "m30", "m35",
            "m40", "mix design", "water cement ratio", "cover", "curing", "workability",
            "slump", "compressive strength", "durability", "concrete design", "pcc", "plain concrete"
        ],
        "applications": ["all RCC structures", "slabs", "beams", "columns", "foundations"],
        "test_standards": ["IS 516", "IS 1199"],
    },
    {
        "id": "IS_10262",
        "code": "IS 10262",
        "title": "Concrete Mix Proportioning — Guidelines",
        "category": "Concrete",
        "description": (
            "Guidelines for proportioning concrete mixes to achieve specified characteristic "
            "compressive strength. Provides systematic approach for design mix selection based "
            "on target mean strength, standard deviation, water-cement ratio, aggregate grading, "
            "and admixture dosage. Essential for producing consistent quality concrete."
        ),
        "keywords": [
            "mix design", "mix proportioning", "concrete mix", "target strength",
            "water cement ratio", "w/c ratio", "aggregate", "trial mix", "design mix",
            "m20 mix", "m25 mix", "m30 mix", "concrete proportion", "grade concrete"
        ],
        "applications": ["all structural concrete mix designs", "ready-mix concrete", "site concrete"],
        "test_standards": ["IS 516", "IS 383"],
    },
    {
        "id": "IS_4926",
        "code": "IS 4926",
        "title": "Ready-Mixed Concrete — Code of Practice",
        "category": "Concrete",
        "description": (
            "Code of practice for the production and delivery of ready-mixed concrete (RMC). "
            "Covers batching plant requirements, mixing, transportation, delivery, acceptance "
            "testing, and quality control for ready-mix concrete supplied from central batching plants."
        ),
        "keywords": [
            "ready mix concrete", "rmc", "batching plant", "transit mixer", "concrete delivery",
            "central mix", "ready mixed", "concrete plant", "volumetric mixer"
        ],
        "applications": ["ready-mix concrete supply", "large construction projects", "high-rise buildings"],
        "test_standards": ["IS 516", "IS 1199", "IS 10262"],
    },
    {
        "id": "IS_9103",
        "code": "IS 9103",
        "title": "Concrete Admixtures — Specification",
        "category": "Concrete",
        "description": (
            "Specification for chemical admixtures used in concrete including plasticizers, "
            "superplasticizers, retarders, accelerators, air-entraining agents, and waterproofing "
            "admixtures. Covers performance requirements, dosage limits, and compatibility "
            "with cement. Essential for high-performance and self-compacting concrete."
        ),
        "keywords": [
            "admixture", "superplasticizer", "plasticizer", "water reducer", "retarder",
            "accelerator", "air entraining", "waterproofing admixture", "chemical admixture",
            "scc", "self compacting concrete", "high performance concrete", "concrete additive"
        ],
        "applications": ["high-performance concrete", "SCC", "pumped concrete", "waterproof concrete"],
        "test_standards": ["IS 516"],
    },
    {
        "id": "IS_516",
        "code": "IS 516",
        "title": "Methods of Tests for Strength of Concrete",
        "category": "Concrete",
        "description": (
            "Standard test methods for determining the compressive, flexural, and tensile strength "
            "of concrete including cube test, beam test, split cylinder test, and core test. "
            "Defines sampling, curing, and testing procedures for fresh and hardened concrete."
        ),
        "keywords": [
            "concrete test", "cube test", "compressive strength test", "flexural strength",
            "split tensile", "core test", "concrete sampling", "strength test", "cube crushing"
        ],
        "applications": ["quality control testing", "acceptance testing", "structural assessment"],
        "test_standards": [],
    },
    {
        "id": "IS_1199",
        "code": "IS 1199",
        "title": "Methods of Sampling and Analysis of Concrete",
        "category": "Concrete",
        "description": (
            "Standard methods for sampling fresh concrete and determining workability by slump test, "
            "compacting factor test, and Vee-Bee consistometer. Also covers methods for analysis "
            "of fresh concrete for cement content and water-cement ratio."
        ),
        "keywords": [
            "slump test", "workability", "fresh concrete", "compacting factor",
            "vee bee", "concrete sampling", "consistency", "slump cone", "flow test"
        ],
        "applications": ["site quality control", "ready-mix acceptance", "mix design validation"],
        "test_standards": [],
    },
    {
        "id": "IS_3812",
        "code": "IS 3812",
        "title": "Pulverised Fuel Ash — Specification (for use as Pozzolana and Admixture in Cement, Cement Mortar and Concrete)",
        "category": "Concrete",
        "description": (
            "Specification for pulverised fuel ash (PFA/fly ash) used as a pozzolanic admixture "
            "in concrete and cement. Covers two grades: Grade 1 for use in concrete and Grade 2 "
            "for use in cement manufacture. Specifies fineness, LOI, lime reactivity, and "
            "chemical composition requirements."
        ),
        "keywords": [
            "fly ash", "pfa", "pulverised fuel ash", "pozzolana", "pozzolanic admixture",
            "blended concrete", "supplementary cementitious", "scm", "thermal power ash"
        ],
        "applications": ["concrete admixture", "cement replacement", "blended concrete"],
        "test_standards": ["IS 1727"],
    },
    {
        "id": "IS_15388",
        "code": "IS 15388",
        "title": "Silica Fume — Specification",
        "category": "Concrete",
        "description": (
            "Specification for silica fume (microsilica) used as supplementary cementitious "
            "material in high-performance concrete. Provides high pozzolanic activity, reduces "
            "permeability, and significantly improves durability and compressive strength. "
            "Used in bridges, marine structures, and ultra-high-strength concrete."
        ),
        "keywords": [
            "silica fume", "microsilica", "high performance concrete", "ultra high strength",
            "hpc", "uhpc", "permeability", "durability", "pozzolan", "supplementary cementitious"
        ],
        "applications": ["high-performance concrete", "marine structures", "bridges", "UHPC"],
        "test_standards": ["IS 1727"],
    },
    {
        "id": "IS_1343",
        "code": "IS 1343",
        "title": "Prestressed Concrete — Code of Practice",
        "category": "Concrete",
        "description": (
            "Code of practice for the design and construction of prestressed concrete structures "
            "including pre-tensioned and post-tensioned systems. Covers materials, prestressing "
            "force, losses, anchorage, grouting, and structural design requirements."
        ),
        "keywords": [
            "prestressed concrete", "psc", "pre-tensioned", "post-tensioned", "prestressing wire",
            "tendon", "anchorage", "grouting", "bridge girder", "long span"
        ],
        "applications": ["bridges", "long-span beams", "railway sleepers", "parking structures"],
        "test_standards": ["IS 516", "IS 1608"],
    },
    {
        "id": "IS_13920",
        "code": "IS 13920",
        "title": "Ductile Design and Detailing of Reinforced Concrete Structures Subjected to Seismic Forces — Code of Practice",
        "category": "Concrete",
        "description": (
            "Code of practice for ductile detailing of reinforced concrete structural members "
            "in seismic zones. Covers confinement reinforcement, lap splices, anchorage length, "
            "beam-column joints, and shear wall detailing to ensure energy dissipation capacity "
            "during earthquakes."
        ),
        "keywords": [
            "seismic", "earthquake", "ductile detailing", "seismic zone", "confinement",
            "beam column joint", "shear wall", "lateral load", "earthquake resistant",
            "rcc seismic", "zone iii", "zone iv", "zone v"
        ],
        "applications": ["buildings in seismic zones", "earthquake-resistant structures"],
        "test_standards": ["IS 456"],
    },

    # ─── AGGREGATES ──────────────────────────────────────────────────────────
    {
        "id": "IS_383",
        "code": "IS 383",
        "title": "Coarse and Fine Aggregates for Concrete — Specification",
        "category": "Aggregates",
        "description": (
            "Specification for coarse and fine aggregates obtained from natural sources and "
            "manufactured aggregates for use in concrete. Covers grading zones, particle shape, "
            "surface texture, deleterious materials, soundness, alkali reactivity, and "
            "mechanical properties. The primary standard for all concrete aggregates."
        ),
        "keywords": [
            "aggregate", "coarse aggregate", "fine aggregate", "sand", "gravel", "crushed stone",
            "granite aggregate", "20mm aggregate", "10mm aggregate", "40mm aggregate",
            "grading zone", "river sand", "m-sand", "manufactured sand", "grit"
        ],
        "applications": ["all concrete mixes", "mortar", "asphalt", "base course"],
        "test_standards": ["IS 2386"],
    },
    {
        "id": "IS_2386",
        "code": "IS 2386 (Parts 1–8)",
        "title": "Methods of Test for Aggregates for Concrete",
        "category": "Aggregates",
        "description": (
            "Test methods for aggregates used in concrete covering: Part 1 (particle size and shape), "
            "Part 2 (estimation of deleterious materials), Part 3 (specific gravity, density, "
            "absorption), Part 4 (mechanical properties including crushing value, impact value, "
            "abrasion value), Part 5 (soundness), Part 6 (mortar-making), Part 7 (alkali-aggregate "
            "reactivity), Part 8 (petrographic examination)."
        ),
        "keywords": [
            "aggregate test", "flakiness index", "elongation index", "crushing value",
            "impact value", "abrasion value", "los angeles", "soundness", "specific gravity",
            "water absorption", "sieve analysis", "deleterious material", "alkali silica"
        ],
        "applications": ["aggregate quality testing", "material acceptance", "mix design"],
        "test_standards": [],
    },
    {
        "id": "IS_515",
        "code": "IS 515",
        "title": "Natural and Manufactured Sand for Masonry Mortars — Specification",
        "category": "Aggregates",
        "description": (
            "Specification for fine aggregates (natural and manufactured sand) used specifically "
            "in masonry mortars for brickwork, blockwork, and plastering. Covers grading, "
            "deleterious material content, and silt content limits."
        ),
        "keywords": [
            "masonry sand", "mortar sand", "plaster sand", "fine aggregate masonry",
            "manufactured sand mortar", "brickwork mortar", "blockwork", "plastering sand"
        ],
        "applications": ["brick masonry mortar", "plaster", "block laying mortar"],
        "test_standards": ["IS 2386"],
    },
    {
        "id": "IS_9142",
        "code": "IS 9142",
        "title": "Artificial Lightweight Aggregates for Concrete Masonry Units — Specification",
        "category": "Aggregates",
        "description": (
            "Specification for lightweight aggregates including sintered fly ash aggregate, "
            "expanded clay, shale, and other artificial lightweight materials used in concrete "
            "masonry units and lightweight concrete for reduced dead load construction."
        ),
        "keywords": [
            "lightweight aggregate", "leca", "expanded clay", "sintered fly ash",
            "lightweight concrete", "light weight block", "low density concrete", "foam concrete"
        ],
        "applications": ["lightweight blocks", "insulating concrete", "reduced dead load construction"],
        "test_standards": ["IS 2386"],
    },

    # ─── BRICKS & MASONRY ────────────────────────────────────────────────────
    {
        "id": "IS_1077",
        "code": "IS 1077",
        "title": "Common Burnt Clay Building Bricks — Specification",
        "category": "Bricks and Masonry",
        "description": (
            "Specification for common burnt clay building bricks used in walls and partitions. "
            "Classifies bricks into classes based on compressive strength (3.5, 5, 7.5, 10, 12.5, "
            "15, 17.5, 20, 25, 35 N/mm²). Covers dimensions, tolerances, water absorption, "
            "efflorescence, and strength requirements."
        ),
        "keywords": [
            "clay brick", "burnt clay brick", "red brick", "building brick", "wall brick",
            "masonry brick", "brick strength", "first class brick", "second class brick",
            "common brick", "modular brick", "non-modular brick"
        ],
        "applications": ["load-bearing walls", "partition walls", "boundary walls"],
        "test_standards": ["IS 3495"],
    },
    {
        "id": "IS_3495",
        "code": "IS 3495 (Parts 1–4)",
        "title": "Methods of Tests for Burnt Clay Building Bricks",
        "category": "Bricks and Masonry",
        "description": (
            "Test methods for burnt clay building bricks: Part 1 (compressive strength), "
            "Part 2 (water absorption), Part 3 (efflorescence), Part 4 (warpage). "
            "Used for quality control and acceptance testing of bricks."
        ),
        "keywords": [
            "brick test", "brick compressive strength", "water absorption brick",
            "efflorescence test", "brick quality test", "warpage", "brick acceptance"
        ],
        "applications": ["brick quality testing", "material acceptance"],
        "test_standards": [],
    },
    {
        "id": "IS_12894",
        "code": "IS 12894",
        "title": "Pulverised Fuel Ash Lime Bricks — Specification",
        "category": "Bricks and Masonry",
        "description": (
            "Specification for fly ash lime bricks manufactured from pulverised fuel ash and "
            "lime as primary raw materials. Provides dimensional consistency, good insulating "
            "properties, and reduced environmental impact. Covers compressive strength, water "
            "absorption, and dimensional requirements."
        ),
        "keywords": [
            "fly ash brick", "fly ash lime brick", "pfa brick", "ash brick",
            "eco brick", "green brick", "sustainable brick", "lime brick"
        ],
        "applications": ["wall construction", "partition walls", "eco-construction"],
        "test_standards": ["IS 3495"],
    },
    {
        "id": "IS_13757",
        "code": "IS 13757",
        "title": "Burnt Clay Fly Ash Building Bricks — Specification",
        "category": "Bricks and Masonry",
        "description": (
            "Specification for building bricks manufactured from a mixture of clay and fly ash "
            "and burnt in kilns. Covers minimum fly ash content (25%), compressive strength, "
            "water absorption, and efflorescence. Better than common clay bricks in dimensional "
            "accuracy and strength."
        ),
        "keywords": [
            "clay fly ash brick", "burnt fly ash brick", "composite brick",
            "fly ash clay brick", "kiln brick", "blended clay brick"
        ],
        "applications": ["general masonry", "load-bearing walls", "boundary walls"],
        "test_standards": ["IS 3495"],
    },
    {
        "id": "IS_2185_1",
        "code": "IS 2185 (Part 1)",
        "title": "Concrete Masonry Units — Specification for Hollow and Solid Concrete Blocks",
        "category": "Bricks and Masonry",
        "description": (
            "Specification for hollow and solid concrete masonry units (CMUs) manufactured from "
            "cement, aggregates, and water. Covers three grades based on compressive strength: "
            "Grade A (7.0 MPa), Grade B (5.0 MPa), Grade C (3.5 MPa). Used for load-bearing "
            "and non-load-bearing walls."
        ),
        "keywords": [
            "concrete block", "cmu", "hollow block", "solid block", "concrete masonry unit",
            "cement block", "block masonry", "load bearing block", "partition block",
            "concrete masonry", "block wall"
        ],
        "applications": ["load-bearing walls", "partition walls", "retaining walls"],
        "test_standards": ["IS 2185"],
    },
    {
        "id": "IS_6441",
        "code": "IS 6441",
        "title": "Autoclaved Cellular Concrete (ACC) Products — Specification",
        "category": "Bricks and Masonry",
        "description": (
            "Specification for autoclaved aerated concrete (AAC) blocks and panels manufactured "
            "by autoclaving a mixture of sand, cement, lime, and aluminium powder. Provides "
            "excellent thermal insulation, lightweight, and fire resistance. Used in partition "
            "and infill walls of framed structures."
        ),
        "keywords": [
            "aac block", "acc block", "autoclaved aerated concrete", "lightweight block",
            "siporex", "ytong", "foam block", "thermal insulation block", "light weight wall",
            "autoclaved cellular", "aerated block"
        ],
        "applications": ["partition walls", "infill walls", "thermal insulation applications"],
        "test_standards": [],
    },

    # ─── TILES & FLOORING ────────────────────────────────────────────────────
    {
        "id": "IS_1237",
        "code": "IS 1237",
        "title": "Cement Concrete Flooring Tiles — Specification",
        "category": "Tiles and Flooring",
        "description": (
            "Specification for cement concrete flooring tiles used in floors and pavements. "
            "Covers dimensions, transverse strength, resistance to wear, and water absorption. "
            "Available in plain and terrazzo finishes."
        ),
        "keywords": [
            "cement tile", "concrete tile", "floor tile", "paving tile", "terrazzo tile",
            "precast floor tile", "cement floor", "outdoor tile", "pavement tile"
        ],
        "applications": ["floors", "footpaths", "pavements", "public areas"],
        "test_standards": [],
    },
    {
        "id": "IS_654",
        "code": "IS 654",
        "title": "Clay Roofing Tiles — Specification",
        "category": "Tiles and Flooring",
        "description": (
            "Specification for burnt clay roofing tiles including Mangalore pattern, country "
            "tiles, and flat tiles. Covers water absorption, permeability, and transverse "
            "breaking strength requirements."
        ),
        "keywords": [
            "roofing tile", "clay roofing tile", "mangalore tile", "country tile",
            "roof tile", "terracotta tile", "clay roof"
        ],
        "applications": ["residential roofing", "traditional construction"],
        "test_standards": [],
    },
    {
        "id": "IS_15622",
        "code": "IS 15622",
        "title": "Ceramic and Vitrified Tiles — Specification",
        "category": "Tiles and Flooring",
        "description": (
            "Specification for ceramic floor and wall tiles and vitrified/porcelain tiles. "
            "Covers dimensions, surface quality, water absorption, modulus of rupture, "
            "scratch hardness, and slip resistance. Classifies tiles by water absorption "
            "into Groups BIa (≤0.5%) to BIII (>10%)."
        ),
        "keywords": [
            "ceramic tile", "vitrified tile", "porcelain tile", "floor tile", "wall tile",
            "glazed tile", "full body tile", "double charge tile", "gvt", "pgvt",
            "bathroom tile", "kitchen tile"
        ],
        "applications": ["flooring", "wall cladding", "bathrooms", "commercial spaces"],
        "test_standards": [],
    },
    {
        "id": "IS_777",
        "code": "IS 777",
        "title": "Glazed Earthenware Tiles — Specification",
        "category": "Tiles and Flooring",
        "description": (
            "Specification for glazed earthenware (ceramic) wall tiles used for internal wall "
            "cladding in kitchens, bathrooms, and wet areas. Covers glaze quality, dimensional "
            "tolerances, and water absorption."
        ),
        "keywords": [
            "glazed tile", "earthenware tile", "wall tile", "ceramic wall tile",
            "bathroom tile", "kitchen tile", "wet area tile", "sanitary tile"
        ],
        "applications": ["bathroom walls", "kitchen walls", "wet area cladding"],
        "test_standards": [],
    },

    # ─── PIPES & PRECAST ─────────────────────────────────────────────────────
    {
        "id": "IS_458",
        "code": "IS 458",
        "title": "Precast Concrete Pipes (with and without Reinforcement) — Specification",
        "category": "Pipes and Precast",
        "description": (
            "Specification for precast concrete pipes used in drainage, sewerage, and culverts. "
            "Covers non-reinforced concrete pipes (NP1 to NP3) and reinforced concrete pipes "
            "(P1 to P3) with various load classes. Specifies dimensions, concrete strength, "
            "three-edge bearing test, and hydrostatic test requirements."
        ),
        "keywords": [
            "concrete pipe", "precast pipe", "rcc pipe", "drainage pipe", "sewer pipe",
            "culvert pipe", "nrm pipe", "np1", "np2", "np3", "stormwater pipe"
        ],
        "applications": ["drainage", "sewerage", "culverts", "stormwater management"],
        "test_standards": ["IS 516"],
    },

    # ─── TESTING & QUALITY ───────────────────────────────────────────────────
    {
        "id": "IS_4031",
        "code": "IS 4031 (Parts 1–15)",
        "title": "Methods of Physical Tests for Hydraulic Cement",
        "category": "Testing",
        "description": (
            "Comprehensive test methods for physical properties of hydraulic cements including "
            "fineness (Part 1–2), soundness (Part 3), setting time (Part 5), compressive "
            "strength (Part 6), tensile strength (Part 8), heat of hydration (Part 9), "
            "and consistency (Part 4). Mandatory testing for all cement grades."
        ),
        "keywords": [
            "cement test", "cement physical test", "fineness test", "soundness test",
            "setting time", "vicat needle", "le chatelier", "cement strength test",
            "heat of hydration test", "cement quality"
        ],
        "applications": ["cement quality control", "acceptance testing", "compliance verification"],
        "test_standards": [],
    },
    {
        "id": "IS_650",
        "code": "IS 650",
        "title": "Standard Sand for Testing of Cement — Specification",
        "category": "Testing",
        "description": (
            "Specification for standard sand (ennore sand) used for testing the strength of "
            "hydraulic cements. Defines grading, silica content, and other properties required "
            "for reproducible cement mortar strength tests."
        ),
        "keywords": [
            "standard sand", "ennore sand", "cement testing sand", "reference sand",
            "silica sand", "test sand", "mortar sand testing"
        ],
        "applications": ["cement strength testing laboratories"],
        "test_standards": [],
    },
]

# ─────────────────────────────────────────────
# TF-IDF + Keyword Boosting Retrieval Engine
# ─────────────────────────────────────────────

def tokenize(text: str) -> List[str]:
    """Lowercase, remove punctuation, split into tokens."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s\-]', ' ', text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 1]


def build_ngrams(tokens: List[str], n: int = 2) -> List[str]:
    """Generate n-grams from token list."""
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def get_query_terms(text: str) -> List[str]:
    """Extract unigrams + bigrams from query."""
    tokens = tokenize(text)
    bigrams = build_ngrams(tokens, 2)
    trigrams = build_ngrams(tokens, 3)
    return tokens + bigrams + trigrams


def build_idf(standards: List[Dict]) -> Dict[str, float]:
    """Compute IDF for all terms across the corpus."""
    N = len(standards)
    df: Dict[str, int] = defaultdict(int)
    for std in standards:
        doc_text = ' '.join([
            std.get('description', ''),
            ' '.join(std.get('keywords', [])),
            std.get('title', ''),
            std.get('category', ''),
            ' '.join(std.get('applications', [])),
        ])
        terms = set(get_query_terms(doc_text))
        for term in terms:
            df[term] += 1
    idf = {}
    for term, freq in df.items():
        idf[term] = math.log((N + 1) / (freq + 1)) + 1.0
    return idf


def score_standard(query_terms: List[str], std: Dict, idf: Dict[str, float]) -> float:
    """Score a single standard against a query using TF-IDF + field weighting + keyword boost."""
    score = 0.0

    field_weights = {
        'keywords': 3.5,
        'title': 2.5,
        'description': 1.5,
        'applications': 2.0,
        'category': 1.0,
        'code': 4.0,
    }

    # Build field term sets with frequencies
    fields = {
        'keywords': ' '.join(std.get('keywords', [])),
        'title': std.get('title', ''),
        'description': std.get('description', ''),
        'applications': ' '.join(std.get('applications', [])),
        'category': std.get('category', ''),
        'code': std.get('code', '') + ' ' + std.get('id', '').replace('_', ' '),
    }

    for field_name, field_text in fields.items():
        field_tokens = get_query_terms(field_text)
        field_term_freq: Dict[str, int] = defaultdict(int)
        for t in field_tokens:
            field_term_freq[t] += 1

        field_len = max(len(field_tokens), 1)
        weight = field_weights[field_name]

        for qt in query_terms:
            if qt in field_term_freq:
                tf = field_term_freq[qt] / field_len
                term_idf = idf.get(qt, 1.0)
                score += weight * tf * term_idf

    # Exact keyword boost: if query term exactly matches a keyword
    keywords_lower = [k.lower() for k in std.get('keywords', [])]
    for qt in query_terms:
        if qt in keywords_lower:
            score += 2.0

    # Category alignment boost
    category_lower = std.get('category', '').lower()
    query_full = ' '.join(query_terms)
    if category_lower in query_full:
        score += 1.5

    return score


def retrieve(query: str, top_k: int = 5) -> List[Dict]:
    """
    Main retrieval function. Returns top-k BIS standards for a query.
    Returns list of dicts with keys: code, title, score, relevance_pct, category, rationale_hint.
    """
    idf = build_idf(BIS_STANDARDS)
    query_terms = get_query_terms(query)

    scored = []
    for std in BIS_STANDARDS:
        score = score_standard(query_terms, std, idf)
        if score > 0:
            scored.append((score, std))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    # Normalize scores to relevance percentage (60–99 range)
    if not top:
        return []

    max_score = top[0][0]
    results = []
    for i, (score, std) in enumerate(top):
        rel_pct = int(60 + (score / max(max_score, 0.001)) * 39)
        rel_pct = min(99, max(60, rel_pct))
        results.append({
            "rank": i + 1,
            "code": std["code"],
            "id": std["id"],
            "title": std["title"],
            "category": std["category"],
            "relevance_score": rel_pct,
            "raw_score": round(score, 4),
            "description": std["description"],
            "keywords": std.get("keywords", [])[:8],
            "applications": std.get("applications", []),
            "test_standards": std.get("test_standards", []),
        })

    return results


if __name__ == "__main__":
    # Quick smoke test
    test_queries = [
        "Ordinary Portland Cement 53 grade for high-rise building",
        "TMT steel bars Fe500D for reinforced concrete",
        "Ready mix concrete M30 for bridge construction",
        "Fly ash bricks for load bearing wall",
        "Coarse aggregate granite 20mm for concrete mix design",
    ]
    idf = build_idf(BIS_STANDARDS)
    for q in test_queries:
        results = retrieve(q, top_k=3)
        print(f"\nQuery: {q}")
        for r in results:
            print(f"  #{r['rank']} {r['code']} — {r['title'][:60]}... ({r['relevance_score']}%)")
