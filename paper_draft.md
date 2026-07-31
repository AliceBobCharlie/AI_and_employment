# Three Dimensions, Not One Index: The Structure of Work Beneath AI Exposure

*Draft. Sections 1 and 2 are written; 3 onward are a running account of what was done and found, to be revised.*

---

## Abstract

We try to answer the questions "what jobs are under greatest AI exposure" and "What other variables can influence AI job displacement". Using O\*NET release 30.2 (894 detailed occupations, 216 requirement ratings and 37 education, training and experience distributions) joined to wages, union coverage, occupational prestige, employment, turnover and self-employment, we apply principal component analysis with varimax rotation and determine dimensionality by resampling rather than by variance explained. Three components are stable and no more: **physical intensity, cognitive load, and person-facing work**. Their interpretations are confirmed against O\*NET's own expert taxonomy, and they are unchanged when every economic variable is removed from the analysis. We propose these three axes, together with union coverage as an orthogonal institutional dimension, as a coordinate system in which occupations can be located and exposure measures compared. Two results follow. Pay tracks cognitive load rather than the manual/mental divide — a median wage gap of $38,520 along cognitive load against $8,940 along physical intensity — and prestige and turnover load on the same axis, so that pay, status and retention are one dimension rather than three. Union coverage, by contrast, is orthogonal to all three axes and to the requirement data generally: an occupation's institutional protection cannot be recovered from any description of its work.

---

## 1. Introduction

### 1.1 Two strategies, two gaps

Research on which jobs artificial intelligence will displace has organised itself around two strategies. The first constructs an occupation-level measure of exposure and ranks occupations along it. The second declines to measure AI and regresses labour-market outcomes on exposure or adoption directly.

Each leaves a gap that the other does not close. An index is a single line, so an occupation whose components point in opposite directions — routine work that AI performs well, alongside judgement that it does not — receives one number that describes neither. And a regression on employment cannot separate what AI is able to do from what employers are permitted to do, since union coverage, policy and firms' willingness to displace all intervene between capability and job loss.

### 1.2 The literature

**Exposure measures.** Work at METR on the length of tasks AI agents can complete establishes that the capability exists and is growing, but is not designed to say which occupations it reaches first. The AI Occupational Exposure index of Felten, Raj and Seamans (2021) was among the earliest occupation-level answers and remains widely used; it weights O\*NET ability ratings by scores describing progress in AI applications, with the link between the two assigned by crowd-sourced matching rather than measured. Eloundou et al. (2024) improved on this by having trained annotators and GPT-4 apply a common rubric to O\*NET task statements and aggregating with importance weights, though the result remains a judgement about tasks rather than a measurement of performance on them, and such ratings are sensitive to which model performs them. More recent measures show the difficulty is structural: the SAFI index of *The AI Skills Shift* is carefully defined but scored by its authors, and the Pew Research Center's analysis introduces institutional and demographic variables of exactly the kind we argue are missing while assigning exposure by the same subjective route. Across this line of work the measurement of the occupation is careful and the measurement of AI is a judgement call.

**Direct regressions.** Brynjolfsson, Chandar and Chen (2025) find a relative employment decline of roughly 16 percent among workers aged 22–25 in the most exposed occupations, concentrated where AI substitutes rather than augments. The counter-evidence is substantial: national studies in Denmark and the United States find no discernible relationship, Humlum and Vestergaard (2025) find essentially zero effect on earnings or hours, and Hartley et al. (2026) report widespread adoption alongside small positive wage effects. Iscenko and Millet (2026) note that exposed occupations are concentrated in interest-rate-sensitive sectors and that postings in them fell before the release of ChatGPT, suggesting part of the decline is macroeconomic. Two years of data is not long enough to separate a trend from a fluctuation, and employment is determined by much besides technology.

### 1.3 What the two gaps cost

**A single number cannot represent an occupation pulled in two directions.** Programmers are the standing example. The repetitive component of the work, and the success of AI at code generation, push the score up; the complex judgement required of senior engineers resists substitution, and demand for it has if anything risen. On one line an occupation must be either exposed or not — it cannot be both — and so the same occupations are argued over indefinitely. We suspect this is part of why exposure measures, despite fitting the available outcome data reasonably well, have not been broadly accepted as descriptions of what is happening.

**Capability is not displacement.** Whether a technically feasible substitution is carried out depends on things unrelated to capability: union coverage and collective bargaining, employers' willingness to displace, and policy. The literature on labour-market power argues that it is the imbalance of power, rather than technology as such, that makes technology threatening to workers, and work on unions and AI documents cases in which collective agreements have shaped what employers may do with the technology. These variables sit between capability and job loss, they vary sharply across occupations, and which occupations currently carry strong external protection is a policy question in its own right. It is not currently answered.

### 1.4 This paper

We approach both gaps from upstream. We do not measure AI. What we analyse is the structure of occupational requirements — the skill side, not the AI side — for reasons set out in Section 2: measuring capability against work requires experimental resources we do not have, and assigning weights by judgement is the practice we are trying to avoid. What remains, and what has not been done, is to establish the structure that every exposure measure is a projection of.

Two commitments follow.

**We assume the impact of AI on an occupation is not one factor but several, occupying a space of more than one dimension, and we try to recover that space rather than impose it.** No variables are selected to represent a concept; the full set of published requirement ratings is used, pruned only on measurable grounds, and the covariance structure determines what the dimensions are. How many there are is settled empirically, by testing which components a random half of the occupations reproduces. The space is then read through its two-dimensional sections, which is what makes a multi-dimensional description usable rather than merely more accurate.

**We collect variables that are not properties of the work — wages, union coverage, prestige, employment, turnover, self-employment — and test which are independent of the skill structure and which are projections of it.** The purpose is not to explain them but to establish which carry information that no description of the work contains, since those are the variables a regression on employment must control for and that policy can act on.

The result is a three-axis description of occupational structure — physical intensity, cognitive load, and person-facing work — together with an institutional dimension orthogonal to all three. We offer it as a model in a specific sense: a coordinate system in which occupations, and the exposure measures built on them, can be located. Its axes are established, their stability tested and their meanings validated against an external classification; what it does not supply is the function by which a position in the space translates into displacement risk. Displacement is some function of the three axes and of institutional protection, and this paper fixes the arguments of that function rather than its coefficients. That is a weaker claim than an exposure index makes, and one that does not expire when the next model is released.

---

## 2. Method

### 2.1 Why the occupation is the unit

Displacement attaches to occupations. A worker is laid off from a job, not from a task and not from a skill; unemployment statistics are collected by occupation; and — decisively for our second question — the institutions that determine whether displacement is permitted attach to occupations as well. Unions organise electricians and actors, licences are issued to physicians and lawyers, professional associations defend the boundaries of an occupation. There is no unit at the task level to which collective bargaining could attach.

This is why we do not work at the task level, as exposure measures built on O\*NET task statements do. Task-level analysis is the finer instrument for technical feasibility and we make no claim against it. But it has no carrier for the institutional dimension, which is half of our question.

It is also why we do not work in skill space. Prior analysis of this data has treated skills as the objects and studied their relations to one another (Section 2.4). That is informative about the structure of skills, and we use it. But no one is made unemployed from a skill, so a description that lives in skill space cannot be brought into contact with the outcome that motivates the question.

### 2.2 Why upstream

Two routes to an exposure measure are available and neither is open to us. The first is to test AI systems against the work — to construct tasks representative of an occupation and measure performance on them. This is the only route that measures rather than judges, and it requires resources we do not have. The second is to assign weights linking capabilities to requirements, which is available to anyone and is the practice whose arbitrariness motivates this paper.

We therefore take neither, and analyse instead what lies upstream of both: the structure of the requirements themselves. This is a real limitation and we state it plainly — **we do not measure exposure to AI, and no result below is evidence about what AI can do.** What the analysis establishes is the space in which any such measurement must be expressed. If two published exposure indices disagree about an occupation, the disagreement can be located in this space; if an index turns out to be a projection onto a single axis, that fact is visible here and not in the index itself.

### 2.3 Data

**Occupational requirements.** O\*NET release 30.2, restricted to occupations with published ratings, giving 894 detailed occupations. Pruning is by measurement, not preference:

- *Importance ratings only.* Skills, Abilities, Knowledge and Work Activities each carry an importance and a level rating. Across elements these correlate at a median of 0.94, and the level matrix is recoverable from the importance matrix at R² ≈ 0.90. Keeping both doubles the column count while adding roughly a tenth of the information, and dilutes every other variable in the matrix in proportion. Level ratings also carry O\*NET's "not relevant" flags, so dropping them removes that missingness. Handel (2009) argues that level is the conceptually richer construct, and we note the disagreement.
- *Work Styles and Interests are dropped* — in this release they are entirely model-generated.
- *Work Values is dropped* — 2008 vintage, 1.1 percent of variance, and predictable from the remaining blocks at R² = 0.75.
- *Job Zones is dropped* — correlated 0.85 with the leading component and predictable from other blocks at R² = 0.89.

This leaves **216 requirement ratings** and the **37 education, training and experience distributions**, all observed and on a single scale.

**Institutional and economic variables.** Wages and employment from OEWS; union coverage from the CPS via the BLS crosswalk; occupational prestige from the Occupational Prestige Ratings project; labour-force exit rate, occupational transfer rate and self-employment share from BLS Employment Projections. Occupational openings is excluded as a size proxy (r = 0.89 with employment), as are the BLS entry thresholds (typical education recoverable from the ETE distributions at R² = 0.82; work experience 85 percent missing).

**Imputation.** Wage gaps are filled by cause. OEWS suppresses annual wages above $239,200; where a high percentile is missing while a low one is present the true value lies above the cap and is filled there, since filling at the median would place a surgeon below their own tenth percentile. Occupations absent from OEWS entirely have no anchor and take the column median. Ratios are recomputed from filled levels. All imputation is confined to one step so that every analysis reads the same numbers.

### 2.4 Prior structure in skill space

Alabdulkareem et al. (2018) analysed the same database with the skills as objects rather than the occupations. They normalised the occupation-by-skill matrix by revealed comparative advantage and binarised it, defined the complementarity of two skills as the minimum of the conditional probabilities that an occupation using one also uses the other, thresholded the resulting network, and found that it separates into two communities — one social and cognitive, one sensory and physical — with occupations then characterised by which community they draw on.

We follow that analysis in taking O\*NET as the description of work and in seeking structure rather than imposing it, and depart from it in two respects. We do not binarise: thresholding at RCA > 1 discards the intermediate values that distinguish occupations differing by degree, and the question of whether the reported structure depends on that step is answerable and worth answering (Section 3.1). And we take occupations rather than skills as the objects, for the reason given in Section 2.1.

The closest methodological precedent is Benzell et al. (2019), who applied principal-component factoring with varimax rotation to O\*NET importance ratings and report eight factors. Their procedure differs from ours in two ways that matter for what is found. Their variable set omits Knowledge and Work Context, and they prune iteratively, discarding any item that loads below a threshold on all factors or above one on more than one factor, until every retained item loads cleanly on a single factor. They observe that routineness does not appear among their factors. An item measuring how demanding work is would load across several requirement domains, and is therefore precisely the kind of item that a simple-structure criterion removes.

### 2.5 Recovering the dimensions

Variables are standardised and reduced by principal component analysis. Wages enter as two columns — a log level and a p90/p10 dispersion — rather than nine collinear percentiles, which would let a single construct claim a component by weight of numbers.

**Dimensionality is determined by stability rather than by variance.** Each candidate component is refitted on 200 bootstrap resamples and 50 random half-splits of the occupations, and matched back to the original by Tucker congruence. A component is retained when the fifth percentile of its split-half congruence exceeds 0.90 — that is, when an independent half of the occupations reproduces the same direction almost exactly. Eigenvalue and scree rules ask how much variance a component explains; this asks whether it is there at all, and the two questions can have different answers.

**Rotation.** Retained components are varimax-rotated so that each axis loads on a limited set of variables and can be read. Each axis is then anchored on a marker variable so its sign is fixed rather than arbitrary, under the convention that the positive pole is the one less amenable to automation. Component scores are computed by rotating standardised component scores, not by using loadings as weights, so that the rotated axes remain uncorrelated.

**Validation.** Naming an axis from its loadings is a judgement on which every substantive claim depends, so we check the names against O\*NET's own Content Model hierarchy, in which experts grouped abilities into cognitive, psychomotor, physical and sensory, and work activities into information input, mental processes, work output and interaction. That grouping took no part in the analysis. We also check that the space is separable in fact — that occupations exist in the off-diagonal cells of each plane — and inspect the occupations at each extreme.

**Orthogonality of the external variables.** For each external variable we ask how much of it the requirement data can predict, by ridge regression with cross-validated R². This is reported alongside the variable's loadings on the rotated axes, since the two answer different questions: the loadings say which direction a variable points, the regression says how much of it the work explains in total. A variable can correlate weakly with every individual axis and still be well predicted by a combination of them.

---

## 3. Results

*(running account; to be tightened)*

### 3.1 Occupations are continuous; skills are grouped

Clustering the 894 occupations gives a best silhouette of 0.238 at k = 2, rising only to 0.316 in the leading principal subspace, far below the 0.5 that indicates separated groups. The same is true of the RCA-binarised matrix (0.234). Occupations do not fall into types under either representation.

Clustering the 161 skill elements does give structure: silhouette 0.456 on the published values, against 0.378 via the RCA-and-minimum-conditional-probability construction. The polarization of skills reported by Alabdulkareem et al. is therefore present in the data and is not produced by binarisation — the binarisation, if anything, obscures it. We note in passing that thresholding the complementarity network at 0.6 retains 19.3 percent of skill pairs, so most of the data is not visible in the published figure.

The two findings are consistent, and their conjunction is the more useful statement: skills come in two families, but occupations draw on them in continuously varying proportions. A dichotomy among skills appears as a continuous axis among occupations, and that axis is the first one we recover.

### 3.2 Three stable dimensions

Of eight candidate components, three survive resampling:

| axis | variance | bootstrap p05 | split-half p05 |
|---|---|---|---|
| R1 physical intensity | 20.9% | 0.997 | 0.982 |
| R2 cognitive load | 18.8% | 0.992 | 0.971 |
| R3 person-facing | 7.8% | 0.986 | 0.956 |
| (R4) | — | 0.788 | 0.688 |

The break at the fourth component is abrupt. Solutions with six to eight factors are common in this literature; our results suggest that beyond the third, a different half of the occupations produces a different answer.

The three account for 47.5 percent of variance. Three things bear on that. The stability tests show that what is excluded contains no reproducible direction, so there is no fourth axis being overlooked. O\*NET's requirement ratings are heavily collinear, so a large share of total variance is not independent information to begin with. And the occupational world plausibly does contain finer structure than three dimensions — but a dimension that operates only within a segment of occupations, or that this instrument does not resolve, will not appear as a global axis. Education and wages, discussed below, are examples of variables that are neither fully aligned with the axes nor independent of them.

**R1, physical intensity.** Positive: millwrights, wind turbine technicians, electricians, firefighters, aircraft mechanics. Negative: poets, proofreaders, literature and philosophy professors, judicial law clerks. The positive pole is unambiguous. The negative pole is defined by absence, and its extreme is symbolic and verbal work.

**R2, cognitive load.** Positive: chief executives, robotics engineers, nuclear engineers, biochemists. Negative: models, dishwashers, crossing guards, fast-food workers, refuse collectors, housekeepers.

**R3, person-facing.** Positive: correctional officer supervisors, police officers, flight attendants, emergency physicians, recreational therapists. Negative: electronics engineers, software developers, mathematicians, computer programmers. The positive pole is not care in the emotional-labour sense; loadings include dealing with angry or aggressive people and with conflict. What is present is direct public contact with responsibility attached.

### 3.3 The names hold against an external classification

Mean loadings by O\*NET's own expert categories:

| expert category | R1 | R2 | R3 |
|---|---|---|---|
| Psychomotor abilities | **0.79** | −0.29 | −0.07 |
| Physical abilities | **0.71** | −0.39 | 0.17 |
| Physical work conditions | **0.57** | −0.10 | 0.04 |
| Cognitive abilities | −0.06 | **0.61** | 0.05 |
| Mental processes | −0.09 | **0.65** | 0.13 |
| Complex problem solving | −0.22 | **0.87** | 0.03 |
| Systems skills | −0.23 | **0.85** | 0.03 |
| Health services | −0.06 | 0.22 | **0.61** |
| Interacting with others | −0.08 | 0.47 | **0.41** |
| Social skills | −0.31 | 0.52 | **0.52** |

The decisive row is cognitive abilities: 0.61 on R2, −0.06 on R1. Cognition is essentially absent from the physical axis. R1 separates work by medium, R2 by difficulty, and the two are empirically independent — established here by a classification that took no part in the analysis.

### 3.4 Pay tracks cognitive load, not medium

Splitting at the median of both axes:

| | median wage | n |
|---|---|---|
| physically intensive, cognitively demanding | 77,730 | 202 |
| physically intensive, routine | 48,310 | 245 |
| physically light, cognitively demanding | 95,770 | 245 |
| physically light, routine | 48,150 | 202 |

The off-diagonal cells are fully populated: millwrights, electricians and aircraft mechanics in one, dishwashers and fast-food workers in the other. The median wage gap along cognitive load is $38,520; along physical intensity, $8,940. A physically demanding but cognitively demanding occupation pays 61 percent more than a physically light but routine one.

The medium is not irrelevant — among cognitively demanding occupations, physically light ones pay about $18,000 more — but it is second-order. What is priced is the difficulty of the work rather than the material it is performed in, and the conventional contrast between brains and brawn merges the two.

This describes an association across occupations. It is not a return to individual ability, and not a causal claim.

### 3.5 Which external variables are dimensions

Ridge regressions of each external variable on the requirement data (cross-validated R²):

| variable | R² |
|---|---|
| occupational prestige | 0.86 |
| median wage | 0.77 |
| union coverage | 0.38 |
| log employment | 0.25 |

Loadings on the rotated axes:

| variable | R1 | R2 | R3 |
|---|---|---|---|
| log wage level | 0.16 | **0.77** | −0.04 |
| prestige (supplementary) | 0.50 | **0.78** | 0.35 |
| labour-force exit rate | 0.07 | **−0.66** | 0.16 |
| occupational transfer rate | −0.16 | **−0.51** | −0.02 |
| union coverage | −0.24 | 0.04 | 0.20 |
| self-employment share | 0.07 | −0.10 | 0.03 |

Pay, prestige and both turnover rates load on the same axis in the same direction. They are not four facts about an occupation but one: cognitive load is simultaneously the wage axis, the status axis and the retention axis. High-load occupations pay more, rank higher, and lose fewer of their workers each year. This is visible only because the axes were derived without reference to any of these variables.

Two variables are largely independent of the requirement data, for different reasons. Employment size is independent trivially — how many people hold an occupation is a fact about the market, not about the work. Union coverage is independent non-trivially: one might expect protection to follow the character of the work, since hazardous and physical occupations have historically been easier to organise, and it does not. Its largest loading on any axis is 0.24 and no requirement block predicts it well. An occupation's institutional protection cannot be recovered from any description of what it requires.

Union coverage nonetheless does not organise the occupational space, because in U.S. data it has too little variance to: most occupations have low coverage. Independence and discriminating power are separate properties and union coverage has the first without the second. This is a fact about American labour institutions rather than a measurement failure, and the same variable in a country with broader coverage might behave differently.

### 3.6 Preparation

The education, training and experience distributions have exactly one stable internal dimension (split-half 0.935; the second reaches 0.862 and is not retained). It runs from low thresholds — no credential, no prior experience, brief training — to high, and moves together across all four scales: schooling, prior experience, on-the-job and in-plant training rise and fall as one. There are no distinguishable routes into an occupation in this data, only more or less preparation.

The dimension correlates with the skill axes at a maximum of 0.54: substantially related to what the work demands, not reducible to it. The residual — a threshold higher or lower than the difficulty of the work would require — is where credentialing and licensing would appear, and is the natural next variable for the institutional side of the question.

### 3.7 Robustness

Refitting the whole procedure under three variable sets and comparing loadings by Tucker congruence: with wages as two columns, with all nine raw wage columns, and with every economic variable removed. Congruence with the reported axes is 1.000 / 1.000 / 0.999 and 1.000 / 1.000 / 0.998. The axes are a property of the requirement data; no decision about wages produced them.

Occupations are weighted equally rather than by employment, since the object of study is the structure of occupations. The wage figures therefore describe the median occupation, not the median worker.

*To add: refit using level ratings in place of importance, addressing Handel's argument; refit on occupations with above-median O\*NET sample size; oblique rotation as a check on the orthogonality assumption; parallel analysis alongside the stability criterion.*

---

## 4. Discussion

*(running account)*

**Relation to the task framework.** The two leading axes correspond to the distinctions that Autor, Levy and Murnane (2003) crossed to organise the task literature: manual against cognitive, routine against non-routine. We recover them without assuming them, which supports the framework and supports the method — an unsupervised procedure reproducing a theoretically motivated structure is doing something other than fitting noise. Where we depart is in the weighting. In the task literature the manual/cognitive distinction carries much of the explanatory load; our wage results place the action almost entirely on the routine/non-routine axis. If that holds, framing exposure in terms of manual versus cognitive work measures the less consequential of the two dimensions.

**The contradictory-occupation problem.** The programmer case that no single index can represent is an ordinary point here: high on cognitive load, low on physical intensity, low on person-facing. What makes it contested is that its position is extreme on two axes at once, and an index that averages them returns a middle value describing no one. Expressed as coordinates the disagreement dissolves into a description — which is not a prediction, but is a prerequisite for one.

**Why exposure measures disagree.** Measures built from patents and engineering capability identify manual and technical occupations as most exposed; measures built from model capability identify analytical and professional ones. If these instruments project onto different axes of the same space, the disagreement is not about magnitude but about which dimension each captures. This is directly testable by projecting published indices onto the three axes, and is the immediate next step.

**The institutional dimension.** Of everything examined, union coverage is the variable that no description of the work recovers. If institutional protection is what determines whether a feasible substitution is carried out, then the quantity that matters most for that question is precisely the one that cannot be read off the work — and in the United States it varies too little across occupations to carry much weight in any statistical model. That is itself a finding about the American case, and a reason to expect the same analysis elsewhere to look different.

---

## 5. Limitations

**We do not measure AI.** No result here is evidence about what AI can or cannot do. The contribution is the structure that such evidence would have to be expressed in.

**O\*NET ratings.** Importance scale points are not defined against an external standard, and the National Academies review concluded that they support comparison between occupations but not statements about absolute requirements; our use is comparative. Evaluations have found rating scales biased toward placing behaviours in professional domains high regardless of difficulty, which would make R2 partly a status measure and the wage result partly circular. That finding concerns the behaviourally anchored level scales, which we do not use, but a weaker form cannot be excluded.

**The negative pole of R1 is heterogeneous.** Its extreme is symbolic and verbal work, but the lower half of the distribution also contains light service work; dishwashers sit on the same side as poets. R1 is better read as physical intensity than as an abstract/embodied contrast, and quadrant interpretations should use extreme groups rather than a median split.

**Cross-sectional.** Everything describes one moment. Nothing identifies a mechanism or a change over time.

**Exploratory sequence.** The pruning decisions were made in response to diagnostics computed on the same data. Each is documented with the statistic that motivated it and the axes are unchanged under the main alternatives, but the sequence was exploratory and is reported as such.

---

## References

*(partial; details to verify)*

Acemoglu, D. and Autor, D. (2011). Skills, tasks and technologies. *Handbook of Labor Economics* 4.

Alabdulkareem, A., Frank, M. R., Sun, L., AlShebli, B., Hidalgo, C. and Rahwan, I. (2018). Unpacking the polarization of workplace skills. *Science Advances* 4(7).

Autor, D., Levy, F. and Murnane, R. (2003). The skill content of recent technological change. *Quarterly Journal of Economics* 118(4).

Benzell, S., Brynjolfsson, E., MacCrory, F. and Westerman, G. (2019). Identifying the multiple skills in skill-biased technical change. MIT IDE working paper.

Brynjolfsson, E., Chandar, B. and Chen, R. (2025). Canaries in the coal mine? Six facts about the recent employment effects of artificial intelligence. Stanford Digital Economy Lab.

Center for American Progress. Unions give workers a voice over how AI affects their jobs.

Deming, D. (2017). The growing importance of social skills in the labor market. *Quarterly Journal of Economics* 132(4).

Eloundou, T., Manning, S., Mishkin, P. and Rock, D. (2024). GPTs are GPTs: labor market impact potential of LLMs. *Science*.

Felten, E., Raj, M. and Seamans, R. (2021). Occupational, industry, and geographic exposure to artificial intelligence. *Strategic Management Journal*.

Handel, M. (2016). The O\*NET content model: strengths and limitations. *Journal for Labour Market Research* 49.

Hartley, J., Jolevski, F., Melo, V. and Moore, B. (2026). [generative AI adoption and labour outcomes].

Humlum, A. and Vestergaard, E. (2025). Large language models, small labor market effects.

Iscenko, Z. and Millet, [.] (2026). Looking for the ladder: is AI impacting entry-level jobs?

Pew Research Center. Which U.S. workers are most exposed to AI on their jobs?

*The AI Skills Shift: mapping skill obsolescence, emergence, and transition pathways in the LLM era.* Preprint.

*Unbalanced labor market power is what makes technology — including AI — threatening to workers.*

Weeden, K. (2002). Why do some occupations pay more than others? *American Journal of Sociology* 108(1).