 # All scales: axes with every rating scale in the input

Script: `supplementary/all_scales.py` (descriptive; prints only).

## Purpose

The paper's matrix has 216 columns: Importance for Skills, Abilities, Knowledge and Work Activities, plus the Context mean (CX) for Work Context. This check asks whether the axes keep their meaning when the matrix also takes in Level ratings and the full Work Context category distributions. The comparison is by meaning, not by stability.

## Matrix

910 occupations × 513 columns. Importance comes from `master_clean.csv`; the other blocks come from `master.csv`, with 2,016 missing cells filled by the column median.

| Block | Content | Columns | Share of variance |
|---|---|---|---|
| IM | Importance: Skills, Abilities, Knowledge, Work Activities | 161 | 40.0% |
| LV | Level: Skills, Abilities, Work Activities (Knowledge Level left out) | 128 | 31.8% |
| CXP | Work Context category percentages, lowest category of each item dropped | 220 | 27.3% |
| CTP | Category percentages for Duration of Typical Work Week and Work Schedules, baseline category dropped | 4 | 1.0% |

- **Why one category is dropped:** each Work Context item's five percentages sum to 100, so one category per item is redundant. The lowest category ("never", "not important at all") is the one dropped. The remaining columns then point the same way as the item's mean, and the poles are not filled with negations such as "never exposed to …".
- **Why Knowledge Level is left out:** it is the least reliable Level block (Section 4.2). O\*NET flags 16.3% of Knowledge Level ratings Recommend Suppress, against 0.2% of Knowledge Importance ratings and 2.9% of Work Activities Level ratings. It also has the highest share of ratings marked Not Relevant, 20%, against 3–16% in the other Level blocks.
- **CTP instead of CT:** Work Schedules (regular, irregular, seasonal) is a nominal item, so its CT mean has no meaning. The CT mean of Duration of Typical Work Week (less than 40 hours, 40, more than 40) cannot tell part-time from long hours. The baseline category of each, 40 hours and regular, is dropped, so the four columns kept are departures from a standard week: less than 40 hours, more than 40 hours, irregular, seasonal.
- **CXP weighting:** every CXP column has its variance weighted by one half. Without this, the many category columns dominate the decomposition. The four CTP columns keep full weight.
- **Method:** columns are z-scored and the CXP columns multiplied by √½. The first six components are varimax-rotated together, not in tiers. The unrotated components explain 30.2, 13.2, 6.3, 3.5, 2.5 and 2.2% of the variance, 57.9% in all.
- **Loadings:** a loading is the correlation between a column and an axis score.
- **Signs:** each axis's larger pole is positive. Signs carry no claim about substitutability.

## Axes

| Axis | Variance | Closest paper axis (r) | Reading |
|---|---|---|---|
| W1 | 22.6% | R2 (+0.94) | judgement |
| W2 | 19.3% | R1 (+0.97) | physical intensity |
| W3 | 5.8% | R3 (+0.88); R6 −0.33, R4 +0.25 | person-facing, leaning clinical |
| W4 | 4.7% | R6 (+0.86); R3 +0.39, R2 +0.23 | management and commerce vs specialist |
| W5 | 2.9% | R4 (−0.92) | on the move vs clinical site |
| W6 | 2.6% | R5 (−0.92) | open-ended vs fixed correct standard |

Below, each pole's strongest columns are listed, with a selection from its ten highest-scoring occupations. The script prints the full lists.

**W1, judgement**
- *+ columns:* Deductive and Inductive Reasoning, Complex Problem Solving, Critical Thinking (IM and LV), all about +0.90.
- *− columns:* standing, repetitive motions and handling objects every day (CXP c5); Trunk and Dynamic Strength.
- *+ occupations:* Physicists, Biochemists, Emergency Medicine Physicians, Neurologists.
- *− occupations:* Models, Landscaping Workers, Dishwashers, Pressers.

**W2, physical intensity**
- *+ columns:* Depth Perception, Operation and Control, Operations Monitoring, Inspecting Equipment.
- *− columns:* sitting every day (CXP c5); speaking and listening; indoors, climate-controlled (CXP c5).
- *+ occupations:* Aircraft Mechanics, Millwrights, Derrick Operators, Commercial Divers.
- *− occupations:* Door-to-Door Sales, Telemarketers, Proofreaders, Writers.

**W3, person-facing**
- *+ columns:* Assisting and Caring for Others, Therapy and Counseling, Psychology; physical proximity, contact and exposure to disease every day (CXP c5); Medicine and Dentistry.
- *− columns:* Design, Engineering and Technology, Programming, Drafting.
- *+ occupations:* Emergency Medicine Physicians, Critical Care Nurses, Nurse Midwives, Police Patrol Officers.
- *− occupations:* Photonics, Chemical and Mechanical Engineers, Mechanical Drafters.

**W4, management and commerce**
- *+ columns:* Monitoring and Controlling Resources, Coordinating the Work of Others, Staffing, Guiding Subordinates, Administration and Management, Selling or Influencing Others.
- *− columns:* weak (|r| ≤ 0.36): middle categories of telephone, contact and work-outcome items; Science.
- *+ occupations:* Spa Managers, Chief Executives, IT Project Managers, Sales Managers.
- *− occupations:* Judicial Law Clerks, Bioinformatics Scientists, Airline Pilots, Proofreaders.

**W5, on the move**
- *+ columns:* Spatial Orientation; enclosed vehicle every day (CXP c5); Night, Peripheral and Glare vision; Transportation; outdoors every day (CXP c5).
- *− columns:* indoors, climate-controlled every day (CXP c5); Medicine and Dentistry; Finger Dexterity; exposure to disease (CXP c5).
- *+ occupations:* Airline Pilots, Fish and Game Wardens, Transit Police, Truck Drivers.
- *− occupations:* Oral Surgeons, Dentists, Prosthodontists, Orthodontists.

**W6, open-ended vs fixed standard**
- *+ columns:* History and Archeology, Fine Arts, Thinking Creatively; lower categories of "repeating same tasks" and "being exact".
- *− columns:* repeating same tasks, being exact, decision frequency and impact, time pressure, all at the top category (CXP c5); Evaluating Compliance with Standards.
- *+ occupations:* Choreographers, Dancers, Fitness Instructors, Actors.
- *− occupations:* Public Safety Telecommunicators, Air Traffic Controllers, Airline Pilots, Typists.

## Observations

- **Same six readings.** All six of the paper's readings come back: each W axis correlates 0.86–0.97 with one paper axis.
- **Judgement is the largest axis.** W1 (22.6%) is larger than physical intensity (19.3%), as on the Level scale. With Importance alone, physical intensity is the larger.
- **Joint rotation puts management on the commercial axis.** W4 is R6 with more management: its top columns are directing and staffing, and it correlates +0.39 with R3. W3 picks up part of R4's clinical content. This matches the joint rotation of the paper's own matrix.
- **Working time loads weakly.** No CTP column reaches |r| = 0.45 on any axis. More than 40 hours goes with judgement (W1, +0.41); less than 40 hours goes against judgement (−0.37) and with person-facing work (W3, +0.32). Irregular schedules go with being on the move (W5, +0.39) and physical intensity (W2, +0.32).
- **The extra columns repeat, rather than add.** Level duplicates Importance: the median correlation of the two scales for the same descriptor is 0.94, so poles list the same descriptor twice. The middle CXP categories ("once a month", "less than half the time") load on the opposite pole as mirror images of the top category; this is most visible on the weak negative pole of W4 and the positive pole of W6.

## Stability (exploratory runs; not computed by this script)

The same matrix was resampled in exploratory runs with 200 split-halves and the paper's tiered rotation.

- **Parallel analysis** keeps 17 components.
- **Components 1–3** reproduce one by one (5th-percentile congruence 0.99, 0.98, 0.96), and component 4 is borderline at 0.90.
- **Components 4–6** are not stable as a subspace: the 5th-percentile RMS cosine is 0.83 and the smallest cosine 0.60 (paper matrix: 0.936 and 0.881).
- **Rotated axes R4, R5 and R6** reach 0.77, 0.69 and 0.86 (paper matrix: 0.92, 0.93, 0.91). R5 is the most fragile: with the two CT means in place of the four CTP columns it fell to 0.58.

The all-scales matrix confirms the meaning of the six axes but resolves the second tier less well than the paper's matrix.

## Run

```
python supplementary/all_scales.py
```

Run after `clean_master.py`, and after `pca_rotated.py` for the correlations with R1–R6. Paths are resolved from the project root.