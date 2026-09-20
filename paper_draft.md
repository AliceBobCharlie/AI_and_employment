# The dimensional structure of occupational space: a coordinate system for AI substitution measures


---

## Abstract

We ask which characteristics of an occupation bear on whether AI can substitute
for human labour, and what other variables stand between substitutability and
observed displacement. Using O\*NET release 30.2 — 894 detailed occupations
described by 216 feature ratings and 37 education, training and experience
distributions — joined to wages, union coverage, occupational prestige,
employment, turnover and self-employment, we apply principal component analysis
with varimax rotation and determine dimensionality by resampling rather than by
variance explained. Three components are stable and no more: **physical
intensity, judgement and person-facing work**. Their interpretations are
confirmed against O\*NET's own expert taxonomy; they are unchanged when every
labour-market variable is removed from the analysis; and they remain almost
uncorrelated when the orthogonality constraint is lifted. We propose these three
axes, together with union coverage as an orthogonal institutional dimension, as
a coordinate system in which occupations can be located and substitution
measures compared. Two results follow. Pay tracks judgement rather than the
manual/mental divide — a median wage gap of $38,520 along judgement against
$8,940 along physical intensity — and prestige and turnover load on the same
axis, so that pay, status and retention are one dimension rather than three.
Union coverage, by contrast, is orthogonal to all three axes and to the
feature data generally: an occupation's institutional protection cannot be
recovered from any description of its work. Applying the coordinate system to
two published substitution measures shows them pointing in opposite directions
along the judgement axis, which accounts for their disagreement without
adjudicating between them.

---

## 1. Introduction

Existing research on which occupations artificial intelligence might replace has generally not treated the question as a multidimensional one. Many studies examine a range of underlying data and then collapse their results into a single index, representing how substitutable a given occupation is by AI relative to others. (A note on terminology: this paper avoids the term "exposure", which leaves unresolved whether what is being measured is substitution or human–AI collaboration. "Substitution" is used throughout, except where another author's measure is named.)

Such an approach overlooks the fact that whether AI can substitute for human labour in a given occupation is determined by several factors at once, and that these factors need not point in the same direction. Software engineering illustrates the difficulty. Parts of the work are highly structured and can be handled by current systems without scaffolding, which makes them readily substitutable. Other parts are context-rich and require planning, and continue to depend on experienced engineers. Whether any particular piece of software engineering work is replaced by AI therefore depends on the balance between two conflicting influences. Reducing the substitutability of software engineers to a single index cannot easily represent this, and studies frequently arrive at opposing conclusions when they are in fact capturing different factors. This is one reason why research on AI substitution remains contested and difficult to reconcile, even where individual measures fit observed data well.

This paper attempts to identify the full set of factors bearing on whether an occupation is substitutable by AI, and to characterise them comprehensively. Against this, the advantage of a one-dimensional index is that it is intuitive, whereas complex models involving many variables are often hard to interpret. To address the first problem while preserving that intuitive quality, the paper seeks to derive a low-dimensional subspace of occupational characteristics, in which an occupation's position along each axis determines where it sits within that space. Two- and three-dimensional slices of this space can then be visualised, so that decision-makers in economic policy can continue to read off the relative position of each occupation directly while also recognising the complexity of the question.

The paper further seeks to identify labour-market variables beyond the characteristics of the work itself, that may impede substitution, such as union density and occupational mobility. These are intended to help address the question of substitution versus collaboration, to account for the gap between measures of substitutability and observed employment outcomes, and to provide a basis for policymakers concerned with employment stability.

The result is a three-axis description of occupational structure — physical intensity, judgement, and person-facing work — together with an institutional dimension orthogonal to all three. We offer it as a model in a specific sense: a coordinate system in which occupations, and the substitution measures built on them, can be located. Its axes are established, their stability tested and their meanings validated against an external classification; what it does not supply is the function by which a position in the space translates into displacement risk. Displacement is some function of the three axes and of institutional protection, and this paper fixes the arguments of that function rather than its coefficients. That is a weaker claim than a substitution index makes, and one that does not expire when the next model is released.

## 2. Method

The most direct way to identify the factors governing whether AI can substitute for human labour in an occupation would be to measure AI performance on tasks representative of that occupation, and to combine those measurements with data on the occupation itself. The most advanced work of this kind is GDPval (Patwardhan et al., 2025), in which practitioners drawn from the occupations concerned both set the tasks and graded the results.

Where such measurement is not available, two practices are common. One is to score AI performance directly on a rating scale; the other is to ask a generative model which tasks it could perform. Both yield occupation-level numbers at low cost, but neither produces a quantity that can readily be verified against an external standard, and results of this kind have been shown to vary with the model performing the rating and with the phrasing of the prompt. We have neither the resources to commission expert assessment on the scale GDPval required, nor any wish to introduce judgements of our own in its place.

We therefore restrict the analysis to the demand side. We take the occupational feature data published in O\*NET, together with a set of labour-market variables attached to occupations, and recover from them the principal factors along which occupations differ. No measurement of AI enters at any point.

Two consequences follow, and we state them plainly.

The first is that this paper contains no evidence about what AI can or cannot do. What it offers is a framework — an account of which variables bear on the question and of the space those variables form — rather than an estimate of substitution risk for any occupation. The function mapping a position in that space to a probability of displacement is not estimated here; the paper fixes the arguments of that function and leaves its coefficients to work that can measure capability.

The second consequence is a compensating one. Capability estimates have a short shelf life and must be revised as systems improve, whereas the structure of occupational features changes far more slowly, and the axes along which occupations differ are likely to remain interpretable across successive revisions of any capability measure. A framework specified on the demand side can therefore accommodate new capability evidence without being rebuilt. We would not claim that occupational requirements are fixed — they are themselves reshaped by technology, and O\*NET revises its ratings accordingly — only that they move on a slower timescale than the capabilities being measured against them.

The cost is equally plain. An analysis conducted on this side of the problem cannot say which occupations are most substitutable at any given moment, and cannot be updated to track a changing frontier. What it can do is establish the terms in which such a judgement would have to be expressed.

A further limitation runs deeper than the absence of capability measurement. The requirements an occupation places on a human worker are not necessarily the requirements it places on an AI system. Capabilities that are distinct for people may be a single capability for a model — O\*NET treats law, medicine and marketing as separate knowledge elements because each is acquired through separate and lengthy human training, whereas for a model trained on a broad corpus these are closer to one capability applied at differing densities of training data. While work that makes almost no demand of a person, such as playing through a game to test it, may lie beyond what current systems can do. There is no reason to expect the dimensions along which human requirements vary to coincide with the dimensions along which machine difficulty varies, and a space recovered from human requirements may therefore organise capability evidence only loosely once such evidence is introduced.

We note that this difficulty is not specific to the present paper. No study to date has recovered the structure of AI capability empirically in the way that occupational features have been recovered here; capability is generally represented either by a single index or by a set of domains specified in advance. Until the capability side has a measured structure of its own, the correspondence between the two spaces cannot be assessed in either direction. We record this as a limitation of the approach and as an open question for the field, rather than as one this paper resolves.

A considerable body of work has examined the structure of the O\*NET occupational space. However, that work all rely on some prior assumptions. Some have fixed the number of dimensions in advance, according to an economic theory to be tested or to the external data the result was to be matched against; the theory is then confirmed, but the structure of the data itself is left unexamined. Some transform the input before analysis — binarising continuous ratings, for instance — which may sharpen or weaken the structure that emerges and introduces a particular bias, even where the validity of the final result is unaffected. Others build weighted aggregates in which the weights are heuristic and cannot be verified against anything, although the resulting indicator may still point in broadly the right direction.

We take an alternative approach. We apply unsupervised analysis, specifically clustering and PCA directly to the published feature ratings, without a prior to be confirmed, in order to recover the structure of the space itself; the analysis that follows is then conducted on the structure so obtained.

## 3. Literature

*To be written. The two studies whose methods this paper departs from directly
are summarised here rather than in the data section, since the departures are
analytical rather than about the data.*

**Structure recovered from the same database.** Alabdulkareem et al. (2018)
analysed O\*NET with the skills as objects rather than the occupations. They
normalised the occupation-by-skill matrix by revealed comparative advantage and
binarised it, defined the complementarity of two skills as the minimum of the
conditional probabilities that an occupation using one also uses the other,
thresholded the resulting network, and found that it separates into two
communities — one social and cognitive, one sensory and physical — with
occupations then characterised by which community they draw on. We follow that
work in taking O\*NET as the description of work and in seeking structure rather
than imposing it, and depart from it in two respects. We do not binarise:
thresholding at RCA > 1 discards the intermediate values that distinguish
occupations differing by degree, and whether the reported structure depends on
that step is answerable (Section 5.2). And we take occupations rather than
skills as the objects, because displacement attaches to occupations: a worker is
laid off from a job, not from a skill, and the institutions that determine
whether displacement is permitted — unions, licences, professional associations
— attach to occupations as well.

**The closest methodological precedent.** Benzell et al. (2019) applied
principal-component factoring with varimax rotation to O\*NET importance ratings
and report eight factors. Their procedure differs from ours in two ways that
matter for what is found. Their variable set omits Knowledge and Work Context,
and they prune iteratively, discarding any item that loads below a threshold on
all factors or above one on more than one factor, until every retained item
loads cleanly on a single factor. They observe that routineness does not appear
among their factors. An item measuring how demanding work is would load across
several feature domains, and is therefore precisely the kind of item that a
simple-structure criterion removes. Neither their analysis nor Alabdulkareem et
al.'s tests whether the reported components are reproducible on an independent
subset of occupations.

## 4. Data

### 4.1 O\*NET occupational features

O\*NET describes each occupation through several blocks of descriptors, collected
from different sources and on different scales. Not all of them bear on whether
AI can substitute for the work, and not all of them are observations of the
occupation. We set out what each block contains, how it might bear on the
question, and what we did with it, so that the selection can be examined rather
than taken on trust.

**Table 4.1 — O\*NET blocks in release 30.2 and their treatment**

| block | what it records | scale used | source | bearing on substitution | decision |
|---|---|---|---|---|---|
| Abilities | enduring attributes of the person that influence performance — cognitive, psychomotor, physical, sensory | Importance | Analyst | direct: what the work demands of a worker is what a system would have to supply | kept (52) |
| Skills | developed capacities that facilitate learning and performance | Importance | Analyst | direct | kept (35) |
| Knowledge | organised bodies of principles and facts | Importance | Incumbent | direct, though what is one domain for a person may not be for a model (§2) | kept (33) |
| Work Activities | generalised activities common across occupations | Importance | Incumbent | direct: closest of the blocks to what is actually done | kept (41) |
| Work Context | physical and social conditions of the work | Context (CX) | Incumbent | indirect but material: whether work is done in person, outdoors, under hazard or under time pressure constrains what can be delegated to a system | kept (55) |
| Work Context | the same items on a 1–3 frequency scale | Category (CT) | Incumbent | duplicative | dropped (2) |
| Education, Training and Experience | percent of respondents in each ordinal category of required education, prior experience, on-the-job and in-plant training | percent distributions | Incumbent | entry thresholds are where credentialing and licensing act, which is institutional rather than technical | kept (37 of 41) |
| Work Values | six dimensions of what an occupation offers a worker | Extent | Analyst, 2008 | weak: describes reward, not requirement | dropped |
| Work Styles | personal characteristics affecting performance | Importance | **AI/Expert (model-generated)** | would be direct if observed | dropped |
| Interests | RIASEC occupational interest profile | Occupational Interest | **Machine Learning/Expert (model-generated)** | weak, and not observed | dropped |
| Job Zones | a single 1–5 ordinal summary of required preparation | ordinal | Analyst | direct in principle | dropped as redundant |

Four blocks are excluded and the reasons differ in kind. Two are excluded
because they are not observations: in release 30.2 the Domain Source field
records Work Styles as AI/Expert and Interests as Machine Learning/Expert, so
both are model outputs rather than measurements of the occupation, and including
them would put a model's judgement inside a matrix intended to hold only
measured features. Two are excluded on evidence, given below.

### 4.2 Labour-market variables

O\*NET describes what an occupation requires. It does not describe the
occupation's position in the labour market, and the question of whether a
technically feasible substitution is actually carried out is partly a question
about that position. We attach six kinds of variable from outside, with a
rationale for each.

**Table 4.2 — Labour-market variables**

| variable | source | why it might bear on substitution |
|---|---|---|
| union coverage rate | CPS 2024, via the BLS National Employment Matrix crosswalk | collective agreements can constrain what an employer may do with a technology irrespective of what it can do |
| median, mean and percentile wages; wage dispersion | OEWS national, May 2024 | the price of the labour being replaced sets the return to replacing it; dispersion within an occupation indicates whether it is one job or several |
| employment | OEWS national, May 2024 | the scale of any displacement, and a control: a measure correlated with occupation size may be measuring size |
| occupational prestige | Occupational Prestige Ratings project | social standing may protect an occupation independently of its technical content; it also serves as a check on whether an axis is a status measure |
| labour-force exit rate; occupational transfer rate | BLS Employment Projections, table 1.10 | how readily workers leave an occupation indicates how easily it sheds labour without dismissals, which is where displacement has been observed first |
| self-employment share | BLS Employment Projections, table 1.2 | the self-employed have no employer to make the substitution decision, so the mechanism differs |

Three further candidates were examined and excluded, each on a measurement
rather than a judgement.

**Occupational openings** correlates 0.89 with employment. It is a size proxy,
and including it would give occupation size two votes.

**Typical education needed for entry**, published by BLS as a categorical
threshold, is recoverable from the ETE distributions at R² = 0.82. The ETE
distributions are finer, so the threshold adds little.

**Work experience needed** is missing for 85 percent of occupations — the field
is populated for 116 of 894 — and cannot support an analysis.

Projections to 2034 were excluded on a different ground. A forecast of an
occupation's growth may already embed a judgement about automation, so relating
it to a structure intended to inform such judgements would be circular.

### 4.3 Variable screening

Four of the blocks — Skills, Abilities, Knowledge and Work Activities — are
published on two scales at once. Importance records how consequential a
descriptor is to the occupation; Level records how much of it the work demands.
Retaining both would be the natural default, and the four blocks would then
occupy 322 columns rather than 161. We retain Importance only, for reasons that
emerged in sequence.

The first is a data problem specific to Level. O\*NET carries a "not relevant"
flag marking descriptors that do not apply to an occupation, and the flag
appears **only on Level rows**: 8.0 percent of Abilities rows and 10.2 percent of
Knowledge rows carry it, and since roughly half the rows in each block are Level
rows, the affected share of Level rows is approximately twice that. The
Importance scale has no counterpart. Retaining Level therefore reintroduces a
missingness problem that Importance does not have.

The second is that this missingness is not arbitrary but semantically tied to
the Importance scale. Every flagged row corresponds to an Importance rating of 2
or below — that is, the Level value is absent exactly where the Importance value
already records that the descriptor barely matters to the occupation. The two
scales are not independently missing; Level goes missing where Importance is
low.

The third followed from testing whether the two scales carry different
information at all. Across the 161 elements, the per-element correlation between
an element's Importance and its Level across occupations has a median of 0.94,
and the full Level matrix is recoverable from the full Importance matrix by
ridge regression at R² ≈ 0.90. Something on the order of a tenth of the
information in Level is not already present in Importance, and that tenth is
concentrated in the elements where the flag makes it least reliable.

Taken together: retaining Level would double the width of four blocks, add
roughly a tenth of the information they already contain, and bring a
missingness problem attached to precisely the least informative cells. It would
also dilute every other variable in the matrix in proportion to the duplication
— a consideration that matters here, since the labour-market variables number fifteen against a feature matrix in the hundreds. We drop Level and record that Handel (2009) argues it is the conceptually richer construct, which we do not dispute; the case above is empirical rather than conceptual.

Four further adjustments are smaller in consequence and are set out together.

**Table 4.3 — Other adjustments to the feature matrix**

| adjustment | evidence | effect |
|---|---|---|
| Work Values dropped | six columns, 1.1 percent of the variance in the leading components, predictable from the remaining blocks at R² = 0.75, and an analyst vintage of 2008 | removes a stale and half-redundant block |
| Job Zones dropped | correlates 0.85 with the leading component, predictable from the other blocks at R² = 0.89, 0.2 percent of variance | a coarser restatement of an axis already present |
| Work Context CT dropped | two items, 0.5 percent of variance, duplicating items already on the CX scale | negligible |
| ETE: highest category of each scale dropped | the categories of each scale sum to 100, so one is linearly determined by the others | removes an exact collinearity; 41 → 37 columns |

One side effect of the Importance-only decision is worth recording, because it
bears on a block that would otherwise look weak. Knowledge carries the highest
rate of both O\*NET reliability flags among the retained blocks — 8.3 percent of
rows recommended for suppression and 10.2 percent marked not relevant. Both
concentrate on Level rows. Once Level is dropped the problem largely goes with
it, and Knowledge becomes as complete as the other three.

### 4.4 Missing values in the occupational features

Missingness has to be described before it is handled, because the treatment
depends on what caused it. We inspected it as an occupation-by-variable matrix
rather than as column totals, which distinguishes two patterns that column
totals would merge.

**Whole-row absence.** O\*NET release 30.2 lists 1,016 O\*NET-SOC codes but
publishes feature ratings for only some of them. For 122 codes there are no
ratings in any block: the row is empty across every descriptor, not sparse. The
composition of those 122 is not arbitrary, and the SOC coding system itself
identifies most of it.

About four fifths — 96 of the 122, or 78.7 percent — are not occupations at all
but artefacts of the classification. The SOC reserves, within each broad group,
a detailed code whose four-digit portion ends in 9 for the "All Other"
residual: `51-3099`, `43-4199` and the like collect the scattered jobs that
belong to a group without warranting a line of their own. A residual bucket is
heterogeneous by construction, so there is no coherent occupation for O\*NET to
survey and it does not survey one. The military major group, `55-xxxx`, is
likewise outside the civilian survey frame in its entirety. A handful of further
cases are individual — legislators, for instance, being elected rather than
employed. Dropping these is not a judgement about which occupations matter but
the absence of anything to measure.

The remaining 26 are ordinary detailed occupations, and what they have in common
is temporal rather than structural: they are codes created or split by the 2018 SOC
revision, for which O\*NET's rolling survey has not yet reached an independent
collection. The 2018 revision separated Data Scientists from the former Computer
Occupations, divided Financial Analysts into Financial and Investment Analysts
and Financial Risk Specialists, broke a general Surgeons code into orthopaedic,
paediatric and other specialisms, and split Emergency Medical Technicians from
Paramedics; Web and Digital Interface Designers, Project Management Specialists,
Medical Records Specialists, Health Information Technologists and Taxi Drivers
arrived or were recoded on the same revision. Their descriptor rows are empty
because collection has not caught up, not because the occupations are marginal.
The data is, in principle, on its way. In practice it has not arrived: these
rows remain empty in release 30.3 as well, so waiting is not an option for the
present analysis, and we exclude them rather than impute values across an entire
feature profile.

This exclusion is the one that carries a cost worth stating plainly, and it is
not a cost of size. Twenty-six occupations out of 920 would barely move the
covariance structure. The cost is that several of them sit close to the centre
of the question this paper is about — data scientists, financial and investment
analysts, web and digital interface designers, project management specialists —
so the analysis set omits a cluster of the symbolic-analytic occupations whose
substitutability is most actively contested, alongside a group of medical
specialisms and emergency services. The axes themselves are estimated from the
894 and would not shift materially were these added. What cannot be done is to
locate any of these occupations in the space, or to make a statement about them
individually. We record the exclusion here and again among the limitations.

**894 detailed occupations remain, and this is the analysis set throughout.**

**Scattered absence.** Among the 894, the feature data is close to complete.
After the Importance-only restriction of Section 4.3, the five feature
blocks — Abilities, Skills, Knowledge, Work Activities and Work Context — have
**no missing cells at all** for any of the 894 occupations. The only gap is in
Education, Training and Experience, where 16 occupations have no rows in any of
the four ETE scales, giving 1.8 percent of cells in that block. Across the whole
253-column feature matrix that is 592 cells out of 226,182, or 0.26 percent.

The quality of what remains is therefore high in a specific sense: the analysis
does not rest on imputed feature values. The sixteen ETE gaps are filled at
the column median, which affects 1.8 percent of occupations on 37 of 253
columns; no other feature cell is imputed.

This completeness is a consequence of the Importance-only decision rather than a
property of O\*NET as published. Had Level been retained and its "not relevant"
flags honoured as missing — which is the correct reading of the flag, since the
value is recorded as not applying rather than as unmeasured — then on the order
of 8 to 10 percent of rows in Abilities and Knowledge would have required either
imputation or the removal of the affected descriptors, concentrated, as Section
4.3 noted, on exactly the descriptors the Importance scale already records as
marginal. The alternative to dropping Level is not a fuller matrix but a matrix
with a tenth of its cells reconstructed.

Two further quality signals are worth recording alongside the missingness, since
O\*NET publishes them and they are rarely reported. The "recommend suppress"
flag marks ratings the programme regards as statistically unreliable, and among
the retained blocks it affects 0.1 percent of Abilities rows, 0.2 percent of
Skills rows, 1.5 percent of Work Activities rows, 2.5 percent of Work Context
rows and 2.3 percent of ETE rows; Knowledge at 8.3 percent is the exception
discussed above, and the flags there sit on Level rows. And the Domain Source
field records who produced each rating: Analyst for Abilities and Skills,
Incumbent for Knowledge, Work Activities, Work Context and ETE. None of the
retained blocks is model-generated, which was the basis for excluding Work
Styles and Interests in Section 4.1.

### 4.5 The final matrix, and whether it is too wide

**Table 4.4 — Final feature matrix**

| block | columns |
|---|---|
| Abilities (Importance) | 52 |
| Work Context (Context) | 55 |
| Work Activities (Importance) | 41 |
| Skills (Importance) | 35 |
| Knowledge (Importance) | 33 |
| Education, training and experience | 37 |
| **total** | **253** |

With the two wage summaries, log employment, union coverage, self-employment and
the two separation rates, the matrix entering the principal component analysis
is 894 occupations by 260 variables.

The ratio of observations to variables is about 3.4, which is low for a
regression but not for a principal component analysis, where the estimand is the
covariance structure and the relevant quantity is whether that structure is
stable rather than whether coefficients are identified. Three properties of the
data and the design bear on it.

The feature ratings are **heavily collinear**, so the effective
dimensionality is far below 260: 76 components are needed to reach 90 percent of
variance, but the first three account for 47.5 percent and the leading structure
is concentrated. Collinearity is a problem for regression coefficients and an
advantage here, since it is what makes a low-dimensional summary possible at
all.

The guard against reading noise as structure is **resampling rather than a rule
of thumb**. Each candidate component is refitted on bootstrap samples and on
random halves of the occupations. A component supported only by the particular
894 occupations in hand would not reproduce on an independent half, and in fact
the fourth and later components do not (Section 5.3). This is a direct test of
the concern that a wide matrix invites, rather than an assumption that it has
been avoided.

Where the width does bite is in the regressions used to ask how much of a labour-market variable the feature data can predict. There, 253 predictors
against 894 observations is genuinely adverse: ordinary least squares does not
merely lose precision but fails, returning large negative cross-validated R²
values, because it fits the training folds exactly and generalises arbitrarily.
Those regressions therefore use ridge, whose penalty is what makes the estimate
meaningful in this shape, with the scaler fitted inside each fold.

### 4.6 Alignment, coverage and imputation of the labour-market variables

The external sources use three coding systems and were collected at different
dates, so their attachment to the 894 occupations is of uneven quality. We set
out the alignment first, then the resulting gaps, then how they were filled,
because the treatment follows from the cause in each case.

#### Alignment

**Exact, on the six-digit SOC.** OEWS, the BLS Employment Projections tables and
the Occupational Prestige Ratings project all key on SOC codes that map one to
one onto O\*NET-SOC codes once the decimal extension is truncated. No information
is lost in the join. Where an O\*NET code has several detailed variants sharing a
six-digit SOC — the specialisms under Physicians, for instance — each variant
receives the same value, which is a property of the published data rather than
of our handling.

**Broadcast, for union coverage.** The CPS occupational classification is
coarser than the SOC: roughly 494 distinct CPS classes back the 894 O\*NET
occupations, so groups of occupations share a coverage rate. This is the weakest
alignment in the dataset. It does not bias the value, but it reduces resolution,
attenuates any correlation involving union coverage, and means the variable
should not be used to make claims about the unionisation of a single occupation.

**Averaged, for prestige.** Several rated job titles can fall under one six-digit
SOC; where they do, the ratings are averaged.

**Dates.** Wages and employment are from May 2024, union coverage from the 2024
CPS, separation rates are projected annual averages for 2024–34, and the
prestige survey is earlier still, while O\*NET revises its blocks on a rotating
schedule so that different occupations carry different vintages. The analysis is
cross-sectional and treats all of this as one period, which is an approximation.

#### Coverage

Unlike the feature data, the labour-market variables have substantial gaps, and
the occupation-by-variable matrix shows them to be structural rather than
scattered: they arrive in whole rows or whole blocks. Of the 894 occupations,
121 are missing at least one labour-market variable, none is missing almost all of them, and 39 are missing only one or two. Three causes account for nearly all of it.

*Whole-row absence from a source.* Twenty-nine occupations have no OEWS record
of any kind — among them laboratory technologists, home health aides, teaching
assistants, assemblers and appraisers. For these there is no anchor from which
to extrapolate a wage.

*Top-coding.* OEWS suppresses annual wages above $239,200. A group of
occupations has its median and upper percentiles suppressed while the tenth and
twenty-fifth percentiles are published: surgeons, anaesthetists, dermatologists
and other physician specialisms, and — through high dispersion rather than high
central earnings — actors, dancers and musicians. This missingness is
informative: it states that the true value lies above the cap.

*Coverage of the source.* Occupational prestige is missing for 53 occupations,
almost all of them detailed management titles the rating project did not cover.

**Table 4.5 — Coverage of the labour-market variables over the 894 analysis occupations**

| variable | coverage | dominant cause of gaps |
|---|---|---|
| union coverage | 100.0% | — |
| self-employment share | 100.0% | blanks denote negligible self-employment |
| labour-force exit rate | 96.9% | absent from the projections table |
| occupational transfer rate | 96.9% | as above |
| employment | 96.8% | absent from OEWS |
| mean wage | 96.4% | absent from OEWS |
| 10th percentile wage | 96.4% | absent from OEWS |
| 25th percentile wage | 96.4% | absent from OEWS |
| median wage | 94.4% | absent from OEWS, or top-coded |
| occupational prestige | 94.1% | not rated by the source project |
| 75th percentile wage | 93.6% | absent, or top-coded |
| 90th percentile wage | 90.8% | absent, or top-coded |

#### Imputation

Because the causes differ, the treatment differs. Where a high wage percentile is
missing while a lower one is present, the occupation is in OEWS and the value has
been suppressed as above the cap; it is filled at the cap, $239,200. Filling it
at the column median would place a surgeon below their own tenth percentile,
which is not a conservative choice but an incorrect one. Ninety-three cells were
filled this way. Where every wage percentile is missing the occupation is absent
from OEWS entirely and has no anchor; these take the column median, 192 cells.
The three wage ratios are then recomputed from the filled levels rather than
filled directly, which keeps them consistent with the levels and no less than
one. Self-employment blanks denote negligible self-employment and are filled at
zero. Remaining gaps, including the unrated prestige scores, take the column
median.

Three considerations bound the risk this introduces. The variables carrying most
of the imputation — the wage percentiles — enter the principal component
analysis as two summary columns rather than nine, so the imputed cells are
diluted before they reach the estimate. The axes are unchanged when every
labour-market variable is removed from the analysis altogether (Section 5.9), so no wage decision can be producing them. And imputation is confined to a single
step, so every analysis reads the same numbers. What imputation cannot be
defended against is a systematic difference between the occupations a source
omits and those it covers: the median fill assumes absence is uninformative, and
for the twenty-nine occupations absent from OEWS we have no evidence either way.

Standardisation is deliberately not applied at this stage. The ridge regressions
of Section 5.7 fit their scaler within training folds only, and storing
standardised values would carry test-fold information into the training set.

# 5. Analysis and results

## 5.1 Standardisation

Every variable is centred and scaled to unit variance before entering any
decomposition. The feature ratings share a scale within a block but not
across blocks — a five-point importance rating and a percentage of respondents
in an education category are not commensurable — and the labour-market variables
range from a proportion to a count in the millions. Without this the leading
component would report which variables were measured in the largest units.

Standardisation is a linear transformation. It moves and rescales a variable but
leaves its shape, its rank order and every correlation it participates in
unchanged, so it is a change of units rather than a modelling choice. This is
worth stating because an alternative normalisation is common in work on this
database: converting the ratings to revealed comparative advantage and
thresholding them at 1, which does alter the data. Thresholding discards the
intermediate values that distinguish occupations differing by degree, and what
survives is a claim about which occupations use a descriptor more than average
rather than how much they use it. Section 5.2 treats that difference as an
empirical question rather than settling it here.

Two variables are transformed before standardising. Employment spans several
orders of magnitude, with a maximum sixty-nine times its median, and enters as a
logarithm. Wages enter as two summaries rather than nine percentiles — the log
of the median as a level, the ninetieth-to-tenth percentile ratio as a
dispersion — because nine near-collinear measurements of one quantity would give
that quantity nine votes against one for every other variable.

## 5.2 Attempts at clustering: do occupations form groups?

Describing occupations by position on continuous axes presumes they do not
instead fall into kinds. We test that rather than assume it, clustering by
k-means for k from 2 to 10 and measuring separation by the silhouette
coefficient, which compares a point's average distance to its own cluster
against its distance to the nearest other. Values near 0.5 and above indicate
separated groups; values near 0.2 indicate a partition imposed on a continuum.

**Occupations do not form groups.** The best silhouette is 0.238 at k = 2 and
falls monotonically thereafter. Repeating the test in the leading ten principal
components, in case distance concentration in 260 dimensions was obscuring
structure, raises it only to 0.316. Repeating it on the RCA-binarised matrix
gives 0.234. Under either representation, and in both the full space and a
reduced one, occupations lie on a continuum.

**Skills do form groups.** The same question put to the 161 skill elements
returns a different answer, which matters because the structure reported by
Alabdulkareem et al. (2018) is a structure among skills rather than among
occupations. Reproducing their construction — binarise at RCA > 1, define the
similarity of two skills as the smaller of the two conditional probabilities
that an occupation using one also uses the other — and clustering the result into
two groups gives a silhouette of 0.378. Replacing that construction with the
correlation of the two skills' published ratings across occupations gives
**0.456**. The two-group structure among skills is therefore present in the
data and is not manufactured by binarisation; if anything the binarisation
obscures it. On Sarle's bimodality coefficient the two similarity distributions
are indistinguishable (0.471 and 0.474) and neither exceeds the 0.555 threshold,
so the bimodality that makes two communities visible in the published network is
not a property we can confirm on either construction. We also record that
thresholding the similarity network at 0.6, as the published figure does,
retains 19.3 percent of skill pairs.

The two findings are consistent, and their conjunction is the more useful
statement: **skills come in two families, but occupations draw on them in
continuously varying proportions.** A dichotomy among skills appears as a
continuous axis among occupations. That axis is the first one recovered below.

## 5.3 The principal components of occupational feature space

The standardised matrix — 894 occupations by 260 variables, of which 253 are
feature columns and seven labour-market — is decomposed by principal component analysis. The unrotated solution places 28.3 percent of variance in the first component and requires 76 components to reach 90 percent, which is what heavy collinearity looks like: a great deal of variance, little of it independent.

How many components to keep is decided by whether they reproduce. Each of the
first eight is refitted on 200 bootstrap resamples of the occupations and on 50
random half-splits, and matched back to the original by Tucker's congruence
coefficient. A component is retained when the fifth percentile of its split-half
congruence exceeds 0.90 — when an independent half of the occupations recovers
the same direction in at least 95 percent of splits. The bootstrap asks whether
a component survives resampling the same occupations; the split-half asks
whether it survives estimation on occupations the original never saw. The second
is stricter and is the criterion used.

**Three components reproduce and no more.**

| component | bootstrap p05 | split-half p05 | verdict |
|---|---|---|---|
| 1 | 0.997 | 0.982 | stable |
| 2 | 0.992 | 0.971 | stable |
| 3 | 0.986 | 0.956 | stable |
| 4 | 0.788 | 0.688 | not stable |
| 5 | 0.784 | 0.658 | not stable |
| 6 | 0.891 | 0.724 | not stable |

The break is abrupt rather than gradual: split-half congruence falls from 0.956
to 0.688 between the third and fourth, and no later component recovers. This
matters for reading the literature, where solutions with six to eight factors on
this database are common. Nothing about those solutions is wrong arithmetically;
the question of whether their later factors would be recovered on a different
sample of occupations has not generally been asked.

The three retained components account for 47.5 percent of variance after
rotation.

## 5.4 The residue

That three axes leave over half the variance unaccounted for invites the
objection that a fourth dimension is being missed. The stability test answers it
directly: what is excluded contains no direction an independent half of the
occupations reproduces.

It does not follow that the residual is noise, and the way a real dimension can
fail this test is worth setting out, because it recurs in Section 5.7.
Occupational licensure is the clearest case. Licensure separates occupations
decisively within medicine, law and the skilled trades, and has no variation at
all across most of the economy. A characteristic of that shape has little
variance globally and no consistent direction when estimated on a random half of
occupations, so it cannot appear as a principal component however real and
however consequential it is. The same holds for a characteristic present
everywhere but weakly differentiating.

This is a property of what a principal component is rather than a defect of the
estimate, and it has a consequence we return to: **a variable that matters for a
question can be invisible to any method that locates structure by maximising
variance, and its absence from the components is not evidence that it is
unimportant.** The three axes should be read as the global structure of the
occupational space, not as an exhaustive description of any one occupation.

## 5.5 What the three axes are

The components as estimated are directions of maximum variance, which is a
criterion about magnitude and not about meaning; they typically load a little on
many variables and are hard to read. Rotation addresses this without changing
anything substantive: it leaves the subspace spanned by the three components
exactly as it was, and alters only which directions within that subspace are
reported, choosing ones on which each variable loads either heavily or near
zero. We apply varimax, which maximises the variance of the squared loadings
within each component.

Two conventions are fixed at this point. Because an eigenvector is defined only
up to sign, each axis is anchored on a marker variable — manual dexterity,
complex problem solving, and assisting and caring for others — and its sign set
so the marker loads positively; without an anchor a small change in the input
can flip an axis and silently invert every statement about it. And occupation
scores are computed by standardising the unrotated scores and then rotating,
rather than by using the loadings as weights: loadings carry a
square-root-of-eigenvalue scaling, and because the eigenvalues differ, using
them as weights yields correlated scores where an orthogonal rotation should
yield uncorrelated ones. The maximum absolute correlation among the resulting
three sets of scores is 0.00.

**R1 (20.9 percent of variance): physical intensity.** The positive pole loads on
reaction time (0.88), depth perception (0.88), inspecting equipment (0.87),
operation and control (0.87), multilimb coordination (0.86), exposure to
hazardous equipment (0.86) and operating vehicles (0.84); its extreme
occupations are millwrights, wind turbine technicians, electricians,
firefighters and aircraft mechanics. The negative pole loads on time spent
sitting (−0.63), indoor environmentally controlled conditions (−0.56), and
speaking, written and oral expression (−0.53 to −0.52); its extremes are poets,
proofreaders, literature and philosophy professors and judicial law clerks.

The two poles are not symmetric — 0.88 against 0.63 — and the asymmetry is
interpretively important. This is an axis of physical capability demand whose
low end is defined by the absence of that demand, not a neutral contrast between
embodied and symbolic media. The negative half of the distribution accordingly
contains light service work as well as verbal work: dishwashers sit on the same
side as poets, not because dishwashing is symbolic but because it makes few
demands on reaction time, depth perception or multilimb coordination. Reading R1
as a medium rather than an intensity would misplace them.

**R2 (18.8 percent): judgement.** The axis runs from judgement-intensive work to
procedure-following work. The positive pole loads on complex problem solving
(0.87), deductive reasoning (0.86), critical thinking (0.86), systems analysis
(0.86), systems evaluation (0.85), judgement and decision making (0.83) and
analysing data or information (0.79); its extremes are chief executives,
robotics engineers, nuclear engineers and biochemists. The negative pole loads on
the labour-force exit rate (−0.66), the two lowest education categories (−0.62,
−0.58), repetitive motions (−0.56) and the occupational transfer rate (−0.51);
its extremes are models, dishwashers, crossing guards, fast-food workers and
housekeepers.

**R3 (7.8 percent): person-facing work.** The positive pole loads on assisting
and caring for others (0.77), psychology (0.73), performing for or working
directly with the public (0.72), therapy and counselling (0.71) and dealing with
unpleasant, angry or discourteous people (0.70); its extremes are correctional
officer supervisors, police officers, flight attendants, emergency physicians
and recreational therapists. The negative pole loads on engineering and
technology (−0.52), design (−0.50), production and processing (−0.42) and
programming (−0.40); its extremes are electronics engineers, software
developers, mathematicians and computer programmers. The positive pole is not
care in the emotional-labour sense — conflict handling loads as heavily as
caring. What its occupations share is direct public contact with responsibility
attached.

**Relation to the task framework.** The two leading axes correspond to the
distinctions Autor, Levy and Murnane (2003) crossed to organise the task
literature: manual against cognitive, routine against non-routine. We recover
them without assuming them, which supports the framework and supports the method
— an unsupervised procedure reproducing a theoretically motivated structure is
doing something other than fitting noise. Where we depart is in the weighting.
In the task literature the manual/cognitive distinction carries much of the
explanatory load; our wage results place the action almost entirely on the
routine/non-routine axis. If that holds, framing substitution in terms of manual
versus cognitive work measures the less consequential of the two dimensions.

**Is the orthogonality real or imposed?** Varimax constrains the axes to remain
uncorrelated, and the claim that they measure different things would be circular
if that constraint were doing the work. We therefore refit with promax, an
oblique rotation which begins from the varimax solution and fits a
transformation to a sharpened target without requiring orthogonality. The
resulting factor correlations are **−0.209** between R1 and R2, **−0.173**
between R1 and R3 and **0.171** between R2 and R3, and the axes are the same
axes: congruence between the varimax and promax solutions is 0.988, 0.995 and
0.995. Orthogonality is therefore close to what the data prefers and not an
artefact of the constraint, which also licenses reporting the variance shares
additively. The implementation was verified on synthetic data with a known
factor correlation of 0.5, which it recovered as 0.48. The signs are themselves
mildly informative: physically intensive work is slightly lower on both other
axes, while those two move slightly together.

**On the name.** We call this axis judgement rather than cognitive load, which
was our first label and which we now think misleading in two ways. "Cognitive
load" has an established meaning in psychology — the burden on working memory
during a task — which is not what the axis measures: a high-speed assembly job
is demanding in that sense and sits at the negative pole here. And "cognitive"
invites reading the axis as knowledge or intelligence, which the loadings do not
support. Among O\*NET's own expert categories, process-oriented basic skills load
at 0.75 and content-oriented basic skills at 0.68, while the knowledge domains
reach only 0.27 to 0.38. Structured knowledge is not what the positive pole is
made of, and much of the work that draws on the most structured knowledge is
also the most procedural.

What separates the two poles is not how much must be known but how much of the
work the worker has to decide: at one end complex problem solving, systems
evaluation and judgement and decision making; at the other repetitive motions,
the lowest education thresholds and high turnover — work whose procedure is
given. The poles are therefore judgement-intensive and procedure-following.

This is also what makes the axis worth naming carefully rather than folding into
the manual/mental contrast, because the two cut across each other. Physical work
that calls for continual on-the-spot decisions sits at the positive pole and is
paid accordingly; symbolic work performed to a given procedure sits at the
negative pole and is not. The wage results of Section 5.6 are a statement about
this axis and not about the medium of the work.

## 5.6 Whether the names are right

Naming an axis from its loadings is a judgement on which every later claim
depends. Three checks are applied, each using information that took no part in
the estimation.

**Against O\*NET's own taxonomy.** The Content Model arranges descriptors in a
hierarchy whose element identifiers encode position: O\*NET's authors group
abilities into cognitive, psychomotor, physical and sensory, and work activities
into information input, mental processes, work output and interaction. Truncating
the identifiers to the third level assigns every element to an expert category,
and the mean loading of each category on each axis can be computed. That
grouping was made for unrelated purposes, so agreement is not circular.

| expert category | n | R1 | R2 | R3 |
|---|---|---|---|---|
| Psychomotor abilities | 10 | **0.79** | −0.29 | −0.07 |
| Physical abilities | 9 | **0.71** | −0.39 | 0.17 |
| Physical work conditions | 30 | **0.57** | −0.10 | 0.04 |
| Work output | 9 | **0.52** | 0.07 | −0.05 |
| Sensory abilities | 12 | **0.48** | 0.11 | 0.05 |
| Complex problem solving | 1 | −0.22 | **0.87** | 0.03 |
| Systems skills | 3 | −0.23 | **0.85** | 0.03 |
| Basic skills, process | 4 | −0.24 | **0.75** | 0.24 |
| Basic skills, content | 6 | −0.37 | **0.68** | 0.09 |
| Mental processes | 10 | −0.09 | **0.65** | 0.13 |
| Cognitive abilities | 21 | −0.06 | **0.61** | 0.05 |
| Health services | 2 | −0.06 | 0.22 | **0.61** |
| Social skills | 6 | −0.31 | 0.52 | **0.52** |
| Interpersonal relationships | 14 | 0.00 | 0.32 | **0.42** |
| Interacting with others | 17 | −0.08 | 0.47 | **0.41** |

The categories sort as the names predict. The decisive row is cognitive
abilities: **0.61 on R2 and −0.06 on R1.** Cognition is essentially absent from
the physical axis, so R1 separates work by physical demand and R2 by something
else entirely, and the two are empirically independent — established here by a
classification that was not used to build them.

**Separability.** A claim that two axes measure different things requires
occupations in the off-diagonal cells of their plane; if every demanding
occupation were also physically light, the two axes would be one. Splitting at
the median of both:

| | n | median wage |
|---|---|---|
| physically intensive, judgement-intensive | 202 | 77,730 |
| physically intensive, routine | 245 | 48,310 |
| physically light, judgement-intensive | 245 | 95,770 |
| physically light, routine | 202 | 48,150 |

The off-diagonal cells are fully populated — millwrights, electricians and
aircraft mechanics in one, dishwashers and fast-food workers in the other. The
axes are separable in fact and not only in construction.

The wage figures fall out of the same table and are reported here rather than
held back. The median wage gap along R2 is **$38,520**; along R1, **$8,940**. A
physically demanding but judgement-intensive occupation pays 61 percent more
than a physically light but routine one. The medium is not irrelevant — among
judgement-intensive occupations, the physically light ones pay about $18,000 more — but it is second order. This is an association across occupations weighted equally, not a return to individual ability and not a causal claim.

**Endpoints.** The occupations at each extreme, listed in Section 5.5, are the
least formal check and the most direct: a name that does not fit its own extreme
cases is wrong whatever the loadings say. All three survive it, with the
qualification about R1's negative pole recorded above.

## 5.7 The labour-market variables

The labour-market variables are not requirements of the work but properties of the occupation's position in the labour market, and the question is whether they carry information the features do not.

A decomposition of the pooled matrix cannot answer this. Principal components
are directions of maximum variance, and with 253 feature columns against
fifteen labour-market ones, heavily correlated among themselves, the leading
components are determined almost entirely by the former; a labour-market variable
can appear only by riding on a feature axis. We confirmed that this is a
property of column counts rather than of the variables, by re-running the same
decomposition with the feature side deliberately widened and narrowed:
whether a labour-market variable appears to emerge changes with how many feature
columns are present while nothing about the variable has changed. Multiple
factor analysis, which normalises each group of variables by its own first
singular value so that no group dominates by size, improves on this but does not
resolve it, because it equalises what a group can contribute without giving a
low-variance variable influence within its group. Both methods answer "does this
variable account for a large share of the joint variation" when the question is
"does this variable carry information the features do not contain", and for
a variable with small variance and an independent direction those have different
answers. This is the situation Section 5.4 anticipated.

The question is therefore put directly by regression. For each labour-market variable
we ask how much of it the feature data predicts, estimating by ridge with
five-fold cross-validation, the penalty chosen within each training fold and the
scaler fitted on training folds only. Ordinary least squares is not usable at 253
predictors against 894 observations: it fits each training fold exactly and
generalises arbitrarily, returning large negative cross-validated R², which we
confirmed before adopting the penalised estimator. This quantity depends on
neither the variable's variance nor the number of feature columns.

| variable | cross-validated R² | reading |
|---|---|---|
| occupational prestige | 0.86 | a restatement of the feature data |
| median wage | 0.77 | largely predictable |
| union coverage | 0.38 | largely independent |
| log employment | 0.25 | largely independent |

The complementary question — not how much is explained but in which direction a
variable points — is answered by correlating it with each rotated axis. The two
are not interchangeable: a variable can correlate weakly with every axis and
still be well predicted by a combination of them, so low correlations are not by
themselves evidence of independence.

| variable | R1 | R2 | R3 |
|---|---|---|---|
| log wage level | −0.16 | **0.77** | −0.04 |
| wage dispersion, p90/p10 | −0.32 | 0.37 | 0.02 |
| prestige (supplementary) | 0.50 | **0.78** | 0.35 |
| labour-force exit rate | −0.06 | **−0.66** | 0.16 |
| occupational transfer rate | 0.16 | **−0.51** | −0.02 |
| union coverage | 0.24 | 0.04 | 0.20 |
| self-employment share | −0.07 | −0.10 | 0.03 |
| log employment | −0.14 | 0.06 | 0.20 |

**Pay, status and retention are one dimension, not three.** Wages, prestige and
both separation rates load on R2 in consistent directions: occupations demanding
more judgement pay more, rank higher and lose fewer of their workers each year to
either exit or transfer. This is visible only because the axes were derived
without reference to any of these variables; had wages helped form them, the
finding would be built in.

**Two variables are independent of the feature data, for different reasons.**
Employment size is independent trivially — how many people hold an occupation is
a fact about the market rather than about the work. Union coverage is
independent non-trivially: one might expect protection to follow the character
of the work, since hazardous and physical occupations have historically been
easier to organise, and it does not. Its largest loading on any axis is 0.24, no
feature block predicts it well, and the full feature matrix reaches only
R² = 0.38. An occupation's institutional protection cannot be recovered from any
description of what it requires.

Union coverage nonetheless does not organise the occupational space, because in
United States data it has too little variance to: the median occupation has 8
percent coverage and the quartiles are 4 and 16 percent, so most of the
distribution sits in a narrow band near zero. Independence and discriminating
power are separate properties and union coverage has the first without the
second. This is a fact about American labour institutions rather than a
measurement failure, and the same variable elsewhere might behave differently.
It is also the concrete instance of the caution in Section 5.4.

We note the resolution with which this variable is measured. The CPS
occupational classification is coarser than the SOC, so roughly 494 distinct
classes back the 894 occupations and groups of them share a value. This
attenuates any correlation involving union coverage and means the variable
should not be used to characterise a single occupation.

## 5.8 The internal structure of preparation

The education, training and experience distributions are analysed on their own
by the same procedure, to ask whether the preparation an occupation demands has
internal structure — whether a route through formal schooling is distinguishable
from a route through on-the-job training — or whether it is a single intensity.

**It is a single intensity.** Exactly one component reproduces (bootstrap 0.982,
split-half 0.935); the second reaches 0.862 on split-half and is not retained. It
accounts for 14.9 percent of variance within the block and runs from low
thresholds — no credential, no prior experience, brief training — to high, moving
together across all four scales, with related work experience contributing 37.5
percent of the axis, on-the-job training 23.2, required education 19.9 and
in-plant training 19.4. There are no distinguishable routes into an occupation
in this data, only more or less preparation.

That single dimension correlates with the feature axes at a maximum of
**0.54**: substantially related to what the work demands, and not reducible to
it. The residual — a threshold higher or lower than the difficulty of the work
would require — is where credentialing and licensing would act, and is the
natural next variable for the institutional side of the question.

## 5.9 Robustness

**The axes are a property of the feature data.** The whole procedure was
refitted with wages as two columns (the reported specification), with all nine
raw percentiles, and with every labour-market variable removed so that only
feature data remains. Congruence with the reported axes is 1.000 / 1.000 /
1.000, 1.000 / 1.000 / 0.998 and 1.000 / 1.000 / 0.999. No decision about wages
produced the axes, and removing the labour-market variables entirely does not change
them.

**Rotation.** Reported in Section 5.5: the oblique refit gives factor
correlations of at most 0.209 and congruence of at least 0.988 with the varimax
solution.

**Weighting.** Occupations are weighted equally throughout, since the object of
study is the structure of occupations rather than the distribution of workers
across them. The wage figures therefore describe the median occupation, not the
median worker.

Two checks remain outstanding: refitting on occupations with above-median O\*NET
sample sizes, using the Occupation Level Metadata file, and a parallel analysis
alongside the stability criterion.

---

## 6. Discussion

### 6.1 Which part of the occupational space is most substitutable

The analysis returns three axes: physical intensity, judgement, and
person-facing work. The number was not chosen. It follows from the resampling
criterion of Section 5.3, which retains a component only where an independent
half of the occupations reproduces it, and the fourth candidate component fails
that test by a wide margin. That the answer happens to be three is convenient
rather than designed: three dimensions can be drawn, and every occupation in the
dataset can be placed in a single figure and located by eye.

**[Figure 6.1 — the 894 occupations in the three-axis space, rotatable, coloured
by union coverage.]**

The first thing the figure shows is what the clustering analysis of Section 5.2
established numerically: the distribution is continuous. Occupations fill the
space rather than gathering into groups, there is no region of concentration,
and no partition of them into types is supported by the data. This matters for
how the rest of the discussion should be read. Any boundary drawn in this space
— including the quadrants used below — is a device for description, not a
division that exists in the labour market.

#### What we cannot establish

The question the figure invites is which direction corresponds to greater
substitutability. We are not able to answer it from data, and it is worth being
precise about why.

Brynjolfsson, Chandar and Chen (2025) find that the employment effects of
generative AI appear first not in dismissals of existing workers but in reduced
hiring of new entrants, and that within firms the employment of workers aged 22
to 25 in the most affected occupations declined relative to that of their older
colleagues. This pattern is itself relevant to the institutional argument of
Section 6.3: an employer who will not dismiss an incumbent may simply not
replace one who leaves, and the constraint that produces this asymmetry is
contractual and reputational rather than technical.

Following that logic, we examined the change in the share of workers aged 22 to
25 within occupations in the American Community Survey between 2022 and 2024.
About 60 percent of the cross-occupational variance in that change is real
rather than sampling noise, so the measure is not simply too noisy to carry any
signal. But its correlations with all three axes are below 0.2 in absolute
value, they change sign between the unweighted and employment-weighted
specifications, and the occupations at the extremes are not coherent: the
largest declines include exercise trainers, social workers and bakers, which no
account of language-model capability would place at the front of the queue.

**[Figure 6.2 — change in the young-worker share, 2022–2024, against each axis.]**

Three things stand between the data and an answer. The alignment between survey
occupation codes and O\*NET is imperfect, and the aggregated codes in the survey
broadcast one value across several occupations. Employment responds to a great
deal besides technology — industrial policy, interest rates, trade, and the
post-pandemic reallocation of labour — and none of these is orthogonal to
position in the occupational space, since the sectors they act on are
themselves concentrated in parts of it. And the period since GPT-4 became
available in 2023 is short: two annual observations cannot distinguish a trend
from a fluctuation. We report the null rather than a weak positive reading of
it, and regard the question as not yet answerable with public data.

#### Our conjecture

What follows is a conjecture assembled from the capability literature. It is not
a result of this paper, and we set it out as a hypothesis the coordinate system
makes precise enough to test once suitable data exists.

**Physical intensity.** Progress in robotics has lagged progress in language
models, and the OECD's capability assessment places the largest remaining gaps
between current systems and occupational requirements in manipulation and
robotic intelligence, describing these as the domains likely to prove most
stubborn. Deployment compounds the gap: a language model reaches a worker
through software that is already installed, whereas a robot requires physical
installation, reconfiguration of the workplace and capital expenditure per site.
In the short term this should make physically intensive occupations less
substitutable than symbolic ones. In the longer term the argument depends
entirely on the trajectory of robotics, on which we take no position.

**Judgement.** The procedural end of this axis should be more
exposed than the end requiring autonomous judgement, which is the standard
reading of the task literature. But the margin is narrowing. Capability
evaluations on unstructured, long-horizon work report rapid improvement —
GDPval finds frontier model performance on real occupational deliverables
improving roughly linearly over time and approaching expert quality on some
tasks — so an advantage that rests on judgement being hard is an advantage with
a shrinking half-life.

**Person-facing work.** Occupations involving direct engagement with people
should be less substitutable than purely technical ones, and the OECD assessment
again finds social interaction among the largest remaining gaps. We would add a
distinction the capability framing does not make: what resists substitution at
this pole is not only the difficulty of the interaction but the requirement that
a person be present and accountable for it, which a capability measure does not
register.

If this conjecture holds — that is, if the positive pole of each axis is the
less substitutable one — then the least substitutable occupations are those
positive on all three, and the most substitutable are those negative on all
three.

**[Table 6.1 — the ten occupations furthest into each of the two corners, with
their coordinates.]**

#### The occupations the corners do not describe

Most occupations are not in a corner, and the interesting ones are those whose
coordinates point in different directions. These are the cases a single index
cannot represent, and they are the reason for building a coordinate system
rather than another index.

The example raised in Section 1 can now be answered. Software engineering sits
near the symbolic pole of physical intensity — almost nothing in the work
requires a body — and on the technical side of the person-facing axis. Both
place it among the more substitutable occupations. But on judgement it is
internally divided in a way its single position on the axis conceals: the work
contains procedural components, such as implementing well-specified
functionality, that sit near the procedural pole, and components requiring
autonomous judgement about architecture and trade-offs that sit near the other.
An occupational average places software engineering in the middle of an axis
along which its constituent work is spread from one end to the other.

The plausible consequence is not that the occupation is replaced or spared, but
that it separates. Work at the procedural end is substitutable on all three
axes at once and has no institutional protection; work at the judgement end
retains a defence on one axis. Entry-level positions, which are composed
disproportionately of the former, would be absorbed first, while senior
positions persist and may become more valuable as the scarce complement to an
abundant capability — the mechanism Autor and Thompson (2025) describe, in which
automation raises wages where it removes the less expert part of a job and
lowers them where it removes the expert part. The outcome is a widening gap
within a single occupational title rather than the disappearance of the title.

This reasoning generalises. Wherever an occupation's internal dispersion along an
axis is large relative to its position on that axis, the occupational average is
a poor description, and the effect of substitution will be to separate the
occupation rather than to move it.

#### Reading the two-dimensional sections

Any two axes can be plotted against each other, giving three planes, and any
subset of occupations can be shown in them. Each plane admits a reading of its
quadrants. We give one as an illustration, and it is the plane on which the
conjecture above has the most to say.

**[Figure 6.3 — physical intensity × judgement, all 894 occupations.]**

**Low physical intensity, low judgement — exposed on both counts.** Work that
demands little of the body and little in the way of decisions has a defence on
neither axis. Under the conjecture this is the most substitutable region of the
space, and it is where routine clerical and data-handling work sits.

**High physical intensity, low judgement — reprieved rather than protected.**
The work is procedural, which offers no defence, but it requires physical
capability that current systems do not have and that would need installed
hardware to supply. The protection is real but contingent: it rests entirely on
the state of robotics and on the capital cost of deployment, and it expires if
either changes. Assemblers, material movers and machine operators sit here.

**Low physical intensity, high judgement — protected on one axis, and unevenly.**
Symbolic work requiring autonomous judgement has a defence, but only on the
judgement axis, and that defence is not distributed evenly within an occupation.
This is the software engineering case set out above: the occupation's position
on the judgement axis is an average over work spread along most of it, and
substitution acts on the lower end first. The expected pattern is not the
removal of the occupation but the erosion of its entry-level positions and a
concentration of employment in the roles where judgement is genuinely required.

**High physical intensity, high judgement — protected on both counts.** Work
that requires both a capable body and autonomous decisions is defended on two
axes at once, and under the conjecture this is the most secure region.
Electricians, aircraft mechanics, millwrights and firefighters sit here. It is
worth noting how poorly this region is described by the conventional contrast
between manual and cognitive work, which would place these occupations on the
manual side and infer substitutability from that alone.

One caution about reading any single plane. A two-dimensional section is a
projection, and it merges occupations that the third axis separates. The exposed
quadrant of this plane, in particular, contains two kinds of work that the plane
cannot distinguish: symbolic routine work such as data entry, and low-skill
in-person service such as dishwashing or food preparation. Both make small
demands of physical capability — the latter requires a body to be present but
not a capable one — and both are procedural. They are nevertheless exposed to
different technologies on different timescales, and what separates them is
position on the person-facing axis. The three planes should be read together
rather than any one taken as the summary.

## 6.2 The relation to published substitution measures

A coordinate system is useful if measures built independently of it can be
located within it. We projected two published measures onto the axes. They
disagree with one another — their occupational scores correlate −0.211 — and the
disagreement resolves into a statement about which axis each is tracking.

Eloundou et al. (2024) label O\*NET task statements according to whether access
to a large language model would halve the time required, and aggregate to
occupations. The OECD (2026) AI Capability Gap Index compares nine capability
domains demanded by an occupation against what current systems can do, and sums
the nine gaps; a small gap denotes high substitutability, so we reverse its sign
for comparability.

**Table 6.2 — Correlations with the axes (unweighted / employment-weighted)**

| measure | physical intensity | judgement | person-facing |
|---|---|---|---|
| Eloundou, human annotators | −0.583 / −0.549 | **+0.318** / +0.214 | 0.041 / −0.002 |
| Eloundou, GPT-4 annotations | −0.388 / −0.345 | 0.012 / 0.023 | −0.183 / −0.352 |
| OECD total, sign reversed | 0.123 / −0.176 | **−0.601** / −0.722 | −0.406 / −0.434 |

The two agree that substitutability rises towards the symbolic pole of physical
intensity. On judgement they point in opposite directions: Eloundou's measure
makes the more demanding occupations more substitutable, the OECD's makes them
less. The conflict is not one of magnitude to be split by taking an average of
the two; the instruments are not estimating the same quantity.

### Where the OECD index loses a dimension

The OECD measure is constructed multidimensionally, and in that respect its
design resembles ours: nine capability domains, each assessed separately against
occupational requirements. It is only at the final step that the nine are summed
into a total. That step is where the dimensionality is lost, and the loss is
visible in our coordinates.

Six of the nine domains — language, social interaction, problem solving,
creativity, metacognition, and knowledge, learning and memory — correlate
positively with physical intensity, between 0.31 and 0.56. The remaining three —
vision, manipulation and robotic intelligence — correlate negatively, between
−0.43 and −0.64. Summing them cancels the axis: the total index correlates 0.12
with physical intensity unweighted and −0.18 weighted, against domain-level
correlations three to five times larger in absolute value. What survives
aggregation is the judgement axis, on which six domains agree in sign. A measure
built as multidimensional therefore arrives, after summation, at close to a
single statement about cognitive demand, with its physical dimension
arithmetically removed.

The asymmetry that produces this is a property of the domain list rather than of
the labour market. Six of nine domains concern cognitive capability and three
concern embodied capability, so an unweighted sum gives the cognitive side twice
the representation. The consequence is not that the physical domains are scored
wrongly but that their contribution is outvoted before it reaches the total.

The nine domains are also less independent than their number suggests. Applied
to the 879 occupations they cover, their own principal components place 55
percent of variance in the first and 22 percent in the second; the participation
ratio gives an effective dimensionality of 2.7. The correlation matrix shows
why: the six cognitive domains correlate with one another between 0.69 and 0.94
— problem solving and metacognition at 0.94 are nearly the same variable — and
the three embodied domains between 0.62 and 0.80, with the two groups negatively
related. Nine domains carry between two and three independent directions. A sum
over nine correlated quantities weights those directions by how many domains
happen to represent each, which is a decision about the taxonomy rather than a
measurement of the occupation.

### A limitation the two measures share with ours

The OECD domains are defined against human capability — the measure is described
as grounded in human abilities, and its domains are drawn from a structured
framework of human cognitive, social and physical capacities. This is the same
assumption we identified in Section 2 as a limitation of our own approach: there
is no guarantee that the dimensions along which human requirements vary are the
dimensions along which machine difficulty varies. Our framework and the OECD's
are exposed to it equally. We note it here not as a criticism of theirs but to
be clear that locating their measure in our space does not resolve the problem
for either.

### Why these two and not others

Two further measures were considered and not used. The GDPval task-level results
are the most direct capability measurement available, but the published release
covers 44 occupations of the 894 and the per-occupation scores are not released
as data. The Anthropic Economic Index records actual usage rather than judgement
and carries an automation-versus-augmentation distinction that is otherwise
scarce, but it is keyed to tasks rather than occupations: in our matching, the
median occupation had a single task in common with it, and occupation-level
ratios computed from one or two tasks degenerate towards zero or one. Results
from it move substantially depending on how tasks are aggregated and how
thinly-covered occupations are handled, which is not a defect of the source but
a limit on what it can support at the occupational level.

## 6.3 How do the labour-market variables act

*[Author's text to follow.]*

---

## References

*(partial; details to verify)*

Acemoglu, D. and Autor, D. (2011). Skills, tasks and technologies. *Handbook of Labor Economics* 4.

Alabdulkareem, A., Frank, M. R., Sun, L., AlShebli, B., Hidalgo, C. and Rahwan, I. (2018). Unpacking the polarization of workplace skills. *Science Advances* 4(7).

Autor, D., Levy, F. and Murnane, R. (2003). The skill content of recent technological change. *Quarterly Journal of Economics* 118(4).

Autor, D. and Thompson, N. (2025). Expertise. NBER working paper.

Benzell, S., Brynjolfsson, E., MacCrory, F. and Westerman, G. (2019). Identifying the multiple skills in skill-biased technical change. MIT IDE working paper.

Brynjolfsson, E., Chandar, B. and Chen, R. (2025). Canaries in the coal mine? Six facts about the recent employment effects of artificial intelligence. Stanford Digital Economy Lab.

Center for American Progress. Unions give workers a voice over how AI affects their jobs.

Deming, D. (2017). The growing importance of social skills in the labor market. *Quarterly Journal of Economics* 132(4).

Eloundou, T., Manning, S., Mishkin, P. and Rock, D. (2024). GPTs are GPTs: labor market impact potential of LLMs. *Science*.

Felten, E., Raj, M. and Seamans, R. (2021). Occupational, industry, and geographic exposure to artificial intelligence. *Strategic Management Journal*.

Handel, M. (2016). The O\*NET content model: strengths and limitations. *Journal for Labour Market Research* 49.

Hartley, J., Jolevski, F., Melo, V. and Moore, B. (2026). [generative AI adoption and labour outcomes].

Humlum, A. and Vestergaard, E. (2025). Large language models, small labor market effects.

Iscenko, Z. and Millet (2026). Looking for the ladder: is AI impacting entry-level jobs?

OECD (2026). The OECD AI Exposure Measure. OECD Artificial Intelligence Papers No. 59.

Patwardhan, T. et al. (2025). GDPval: evaluating AI model performance on real-world economically valuable tasks.

Pew Research Center. Which U.S. workers are most exposed to AI on their jobs?

*The AI Skills Shift: mapping skill obsolescence, emergence, and transition pathways in the LLM era.* Preprint.

*Unbalanced labor market power is what makes technology — including AI — threatening to workers.*

Weeden, K. (2002). Why do some occupations pay more than others? *American Journal of Sociology* 108(1).