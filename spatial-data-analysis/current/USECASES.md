
Moving away from geoscience (soil, mining, forestry, meteorology) is a fantastic decision. Classical spatial textbooks lean so heavily on 1960s mining and agriculture that modern statistics students assume spatial statistics is only for agronomists and geologists.

Here are **five modern, high-impact domains** that statistical students actually care about, along with how the three data types map onto them:

---

### 1. Urban Economics & Tech (Real Estate & Mobility)
*   **Continuous (Geostatistical):** **Apartment rental prices or land value surfaces.** If an apartment rents for \$3,000/month, the apartment building 50 feet away is not an independent draw from a Gaussian distribution—it will rent for almost the exact same price.
*   **Areal (Lattice):** **Ride-share surge pricing or median household income by census tract.** Adjacent census tracts share transit lines, school quality, and commercial corridors.
*   **Point Patterns:** **Locations of EV charging stations, coffee shops, or Uber/Lyft pickups.** Locations aren't random; coffee shops cluster near transit hubs, while bank branches deliberately space themselves out (inhibition).

### 2. Modern Epidemiology & Public Health
*   **Continuous (Geostatistical):** **Lead concentration in municipal tap water** or **decibel noise pollution across a city.** Two houses on the same residential street share the same water main pipes and traffic exposure.
*   **Areal (Lattice):** **Opioid overdose death rates or vaccination rates across hospital service areas.** Contiguous health districts share demographic vulnerabilities and policy environments.
*   **Point Patterns:** **Locations of sudden cardiac arrest events or rare pediatric cancer cases.** Testing whether disease cases cluster around an industrial chemical plant compared to the underlying population.

### 3. Cellular Biology & Spatial Omics (Biostatistics)
*   **Continuous (Geostatistical):** **Protein concentration or oxygenation across a tumor biopsy slice.** Chemical gradients diffuse smoothly across cellular tissue.
*   **Areal (Lattice):** **Spatial transcriptomics (gene expression binned into microscopic hexagonal tiles on a slide).** Expression levels in neighboring tissue tiles are highly correlated.
*   **Point Patterns:** **Tumor-infiltrating lymphocytes (immune cells) on a histology slide.** Do immune cells swarm together (cluster) around cancer cells, or do cancer cells establish an exclusion zone (inhibition)?

### 4. Criminology & Public Safety
*   **Continuous (Geostatistical):** **Acoustic gunfire detection sensor response times** across a metropolitan grid.
*   **Areal (Lattice):** **Violent crime rates across police precincts or voting wards.** Crime spillovers cross administrative borders—police presence on one side pushes activity to the adjacent precinct.
*   **Point Patterns:** **Locations of 911 calls, bicycle crashes, or street robberies.** Crashes cluster at specific street intersections with poor sightlines, rather than being uniformly distributed.

### 5. Sports Analytics (Spatial Court/Pitch Data)
*   **Continuous (Geostatistical):** **Expected Goal Value (xG) surface in soccer or shot efficiency surfaces in basketball.** The probability of scoring from $(x, y)$ coordinates changes smoothly as a player moves around the 3-point line.
*   **Areal (Lattice):** **Defensive coverage zones (Voronoi pitch partitioning).** Players divide the court into interdependent territories.
*   **Point Patterns:** **Shot charts (e.g., Stephen Curry’s shot attempts).** Do shot attempts cluster around the high-efficiency corners and arc, exhibiting spatial attraction to the 3-point line?

---

### How That Opening Paragraph Reads (Rewritten Without Geosciences)

Here is how your opening text sounds when rewritten using **urban economics, public health, and human activity**:

> *"Most statistics courses rest on one quiet assumption: the rows of a data set are independent. Each observation is a fresh draw, unrelated to the one before it. That assumption is what makes $\sigma^2/n$ the variance of a sample mean, what makes regression standard errors correct, and what makes a $p$-value mean what it says.*
>
> *Spatial data breaks that assumption in a predictable way. Two apartments on the same street share school districts, transit access, and noise; their rents are more alike than two apartments across town. Two neighboring census tracts share economic conditions, hospital networks, and demographics. Pedestrian traffic accidents cluster at dangerous intersections rather than scattering at random.* 
>
> *When observations near each other are related, the familiar formulas still produce numbers, but the numbers overstate how much the data know. A sample of 500 homes in the same subdivision does not give you 500 independent pieces of information about the housing market; in reality, you might only have the equivalent of 30."*

### Why This Works Better for the B+ Student
* **Zero Dirt / Zero Rocks:** No soil cores, no boreholes, no timber stands.
* **Immediately Relatable:** Every student understands why two apartments on the same block have correlated rents, or why traffic accidents cluster at an intersection.
* **Sets up the Math:** It makes the concept of **Effective Sample Size ($n_{\text{eff}} < n$)** in Chapter 1 feel intuitive and obvious before they ever see the covariance matrix $\boldsymbol{\Sigma}$.



Published research actively applies spatial statistics to supply chains, logistics, and reverse-flow networks across geostatistical, areal, and point-pattern frameworks:

### 1. Reverse Logistics & Continuous Geostatistical Fields

In circular economy and reverse supply chains, predicting product return flows across disparate locations presents a severe spatial autocorrelation challenge. Lv and Du (2021) addressed this by treating waste electrical and electronic equipment (WEEE) collection as a continuous spatial random field. By verifying the second-order stationary hypothesis and estimating experimental variograms across recovery locations, they deployed ordinary kriging models to interpolate and predict collection volumes at unmonitored sites, reducing vehicle dispatch uncertainty.

### 2. Regional Logistics Capacity & Spatial Econometrics (Areal / Lattice Data)

To evaluate how logistics infrastructure generates economic externalities across administrative boundaries, spatial econometric modeling is standard practice. Fanghu et al. (2024) used global and local Moran’s $I$ alongside Spatial Durbin Models (SDM) across provinces in the Yangtze River Economic Belt. Their findings showed that regional logistics capability is not isolated within provincial borders; infrastructure investments in transportation and storage generate statistically significant spatial spillover effects in contiguous economic zones. Similarly, Tsui et al. (2022) applied global and local Moran’s $I$ to evaluate geographic clustering in circular material flows and industrial waste reuse in the Netherlands, pinpointing localized industrial symbiosis hotspots.

### 3. Warehouse Siting & Logistics Agglomeration (Point Patterns)

Analyzing firm entry, warehouse clustering, and logistics infrastructure often requires spatial point pattern techniques to test for spatial dependence rather than assuming complete spatial randomness. Wang et al. (2025) examined the micro-location patterns of logistics businesses and warehousing across Texas metropolitan areas, modeling the spatial concentration and economic interactions between e-commerce distribution nodes and freight corridors. Earlier foundational literature in operational research, such as Klose and Drexl (2005), systematically classified how spatial discrete-choice and continuous facility location formulations interact with distributed customer demand points in two-stage distribution networks.

---

### References

Fanghu, L., Yinnan, H., & Biao, W. (2024). Analysis of logistics capacity, influencing factors and spatial spillover effect in Yangtze River Economic Belt. *PLOS ONE*, *19*(5), e0303200. [https://doi.org/10.1371/journal.pone.0303200](https://doi.org/10.1371/journal.pone.0303200)
Cited by: 14

Klose, A., & Drexl, A. (2005). Facility location models for distribution system design. *European Journal of Operational Research*, *162*(1), 4–29. [https://doi.org/10.1016/j.ejor.2003.10.031](https://doi.org/10.1016/j.ejor.2003.10.031)
Cited by: 1682

Lv, J., & Du, S. (2021). Kriging Method-Based Return Prediction of Waste Electrical and Electronic Equipment in Reverse Logistics. *Applied Sciences*, *11*(8), 3536. [https://doi.org/10.3390/app11083536](https://doi.org/10.3390/app11083536)
Cited by: 10

Tsui, T., Derumigny, A., Peck, D., van Timmeren, A., & Wandl, A. (2022). Spatial clustering of waste reuse in a circular economy: A spatial autocorrelation analysis on locations of waste reuse in the Netherlands using global and local Moran’s I. *Frontiers in Built Environment*, *8*, Article 954642. [https://doi.org/10.3389/fbuil.2022.954642](https://doi.org/10.3389/fbuil.2022.954642)
Cited by: 49

Wang, K., Lim, G. J., Race, B., Zhang, Y., Gao, L., & Qiao, F. (2025). Examining spatial patterns and economic interactions of logistics activities across three Texas metropolitan areas. *Journal of Transport Geography*, *123*, 104079. [https://doi.org/10.1016/j.jtrangeo.2024.104079](https://doi.org/10.1016/j.jtrangeo.2024.104079)
Cited by: 4



In mainstream economics, **spatial econometrics** addresses a major econometric failure: the assumption that observational units (countries, states, firms, workers) are independently distributed. When cross-sectional units engage in trade, migration, or strategic game-theoretic interactions, standard OLS produces biased or inefficient estimates.

Spatial econometrics models this dependence explicitly using a spatial weight matrix $W$, where $W_{ij}$ defines connectivity (geographic contiguity, inverse distance, travel times, or trade volume).

Economists apply spatial econometrics across four major subfields:

---

### 1. Public Finance & Political Economy: Strategic Interaction & Yardstick Competition

Local governments do not set tax rates or public goods in a vacuum; they compete strategically for mobile capital (tax competition) and voters judge incumbents against neighboring leaders (yardstick competition).

* **The Empirical Challenge:** If City $i$ sets property tax $\tau_i$, its choice depends endogenously on the vector of neighbors' taxes $\sum_j W_{ij} \tau_j$ (the spatial lag $W\tau$). Because $i$ influences $j$ while $j$ simultaneously influences $i$, OLS suffers from reflection/simultaneity bias.
* **Model Specification:**

$$\tau = \rho W \tau + X\beta + \varepsilon$$



This Spatial Autoregressive (SAR) / Spatial Lag model estimates $\rho$, the slope of the local government's **reaction function**. A positive $\rho$ proves strategic "tax mimicking".
* **Key Foundational Literature:**
* **Besley & Case (1995)** — *Incumbent Behavior: Vote-Seeking, Tax-Setting, and Yardstick Competition* (*American Economic Review*). Proved that voters use neighboring tax changes as an informational benchmark to evaluate incumbent performance, forcing governors to align tax policies with surrounding states.
* **Brueckner (2003)** — *Strategic Interaction Among Private and Public Agents: Spatial Econometrics in Modern Public Economics* (*International Regional Science Review*). Formalized how local spending and tax-rate setting can be structurally estimated using spatial lag instruments.



---

### 2. Innovation Economics & Growth: Knowledge Spillovers & Agglomeration

New Growth Theory (Romer, Lucas) assumes technological know-how diffuses as a non-rival good, but economic geography shows that idea diffusion degrades with physical distance.

* **The Empirical Challenge:** Patents or total factor productivity (TFP) in a region depend not only on its own local corporate R&D spending, but also on R&D conducted in neighboring regions.
* **Model Specification:** The **Spatial Durbin Model (SDM)** is favored here because it nests both endogenous spillovers ($WY$) and exogenous technology spillovers ($WX$):

$$Y = \rho W Y + X\beta + WX\theta + \varepsilon$$



Using matrix inversion $(I - \rho W)^{-1}$, researchers partition the marginal effect of an R&D tax credit into **Direct Effects** (local productivity growth) and **Indirect / Spillover Effects** (productivity gains leaked into neighboring regions).
* **Key Foundational Literature:**
* **Anselin, Varga, & Acs (1997)** — *Local Geographic Spillovers between University Research and High Technology Innovations* (*Journal of Urban Economics*). Demonstrated that university basic research creates patent spillovers that attenuate sharply beyond a 50-mile radius, formalizing the econometric spatial lag test for regional innovation systems.
* **LeSage & Pace (2009)** — *Introduction to Spatial Econometrics*. Established the mathematical framework for decomposing partial derivatives into direct, indirect, and total spatial impacts in empirical economics.



---

### 3. International Trade & Macroeconomics: Spatial Gravity & Third-Country Effects

Standard trade gravity models predict bilateral trade flows between Country $i$ and Country $j$ based on their GDPs and bilateral transport cost. However, traditional gravity violates independence if trade with Country $k$ creates trade diversion or intermediate input linkages.

* **The Empirical Challenge:** Known as "multilateral resistance" (Anderson & van Wincoop) or third-country effects: a trade agreement signed between the US and Mexico alters Mexican exports to Canada.
* **Model Specification:** **Spatial Matrix $W$ as Trade-Cost or Input-Output Matrices**: Rather than a physical boundary border matrix, economists use economic distance:

$$W_{ij} = \frac{\text{Trade}_{ij}}{\sum_k \text{Trade}_{ik}} \quad \text{or} \quad W_{ij} = \text{Input-Output Linkage}_{ij}$$


* **Key Foundational Literature:**
* **Blonigen, Davies, Waddell, & Nieswiadomy (2007)** — *FDI in Space: Spatial Autoregressive Models of Fine-Grained Foreign Direct Investment* (*European Economic Review*). Showed that multinational outbound FDI exhibits spatial dependence: investments into one European country serve as an export platform, creating spatial complement or substitution effects in neighboring host markets.
* **Ertur & Koch (2007)** — *Growth, Technological Interdependence and Spatial Externalities: Theory and Evidence* (*Journal of Applied Econometrics*). Integrated the Solow growth model with spatial econometric autoregressive panels, showing that capital accumulation spillovers across international trading networks explain cross-country income divergence.



---

### 4. Labor Economics & Spatial Equilibrium: Regional Sorting & Place-Based Policies

Evaluating federal interventions (e.g., enterprise zones, rural infrastructure grants, minimum wage hikes) requires spatial econometrics to capture leakage.

* **The Empirical Challenge:** If County $A$ raises its minimum wage, workers from neighboring County $B$ commute across borders to take those jobs, while businesses may relocate across the boundary line. Treating counties as independent control/treatment observations violates the Stable Unit Treatment Value Assumption (SUTVA).
* **Model Specification:** Spatial Error Models (SEM) account for spatial autocorrelation in the unobserved shocks (such as localized macroeconomic downturns or weather shocks):

$$Y = X\beta + u, \quad u = \lambda W u + \varepsilon$$



This prevents standard errors from being artificially deflated, which otherwise produces widespread false-positive policy significance.
* **Key Foundational Literature:**
* **Redding & Rossi-Hansberg (2017)** — *Quantitative Spatial Economics* (*Annual Review of Economics*). Synthesizes how microdata (commuting flows, cell phone location pings) calibrate spatial general equilibrium models with congestion costs and labor spillovers.



---

### Summary of Core Econometric Specifications

| Model Class | Mathematical Formula | Economic Interpretation |
| --- | --- | --- |
| **SAR** (Spatial Lag) | $Y = \rho W Y + X\beta + \varepsilon$ | **Strategic interaction / Game theory:** Agent $i$'s decision directly mirrors or reacts to agent $j$'s action (e.g., tax mimicking, pricing wars). |
| **SEM** (Spatial Error) | $Y = X\beta + (I - \lambda W)^{-1}\varepsilon$ | **Omitted regional shocks:** Unobserved covariates (e.g., regional culture, weather, common labor supply) correlate across space. |
| **SDM** (Spatial Durbin) | $Y = \rho W Y + X\beta + WX\theta + \varepsilon$ | **Spillovers & Externalities:** Outcomes depend on own attributes, peer outcomes, *and* neighbors' investments (e.g., R&D, infrastructure). |


### 1. Public Finance & Strategic Interaction (Yardstick Competition & Tax Mimicking)

* **Besley, T., & Case, A. (1995).** Incumbent behavior: Vote-seeking, tax-setting, and yardstick competition. *The American Economic Review*, *85*(1), 25–45. [suspicious link removed]
* **Core empirical application:** Evaluates how U.S. governors set state tax rates. Using a spatial autoregressive (SAR) framework, the authors demonstrate that voters evaluate their incumbent's tax decisions by benchmarking them against contiguous neighboring states, forcing politicians into strategic "yardstick" tax competition.


* **Brueckner, J. K. (2003).** Strategic interaction among private and public agents: Spatial econometrics in modern public economics. *International Regional Science Review*, *26*(2), 176–188. [https://doi.org/10.1177/0160017602250974](https://www.google.com/search?q=https://doi.org/10.1177/0160017602250974)
* **Core empirical application:** Establishes the standard theoretical and empirical toolkit for structural reaction functions in economics, laying out how to estimate spatial lag parameters when modeling local public goods provision, zoning regulations, and welfare benefit competition.



---

### 2. Innovation, Knowledge Spillovers, & R&D Agglomeration

* **Anselin, L., Varga, A., & Acs, Z. (1997).** Local geographic spillovers between university research and high technology innovations. *Journal of Urban Economics*, *42*(3), 422–448. [https://doi.org/10.1006/juec.1997.2032](https://doi.org/10.1006/juec.1997.2032)
* **Core empirical application:** Investigates whether university R&D leaks into commercial high-tech patents across U.S. Metropolitan Statistical Areas (MSAs). Employs spatial lag and spatial error models with distance-band spatial weight matrices ($W$), showing that academic knowledge spillovers extend up to a 50-mile radius before rapidly dissipating.


* **LeSage, J. P., & Pace, R. K. (2009).** *Introduction to Spatial Econometrics*. CRC Press / Taylor & Francis. [https://doi.org/10.1201/9781420064254](https://doi.org/10.1201/9781420064254)
* **Core empirical application:** The foundational methodological text for applied economists. It formalizes the Spatial Durbin Model (SDM) and provides the mathematical derivation for scalar summary measures of **Direct**, **Indirect (Spillover)**, and **Total Impacts**, preventing the misinterpretation of spatial regression coefficients as simple marginal effects.



---

### 3. International Trade, FDI, & Macroeconomic Growth

* **Blonigen, B. A., Davies, R. B., Waddell, G. R., & Naughton, H. T. (2007).** FDI in space: Spatial autoregressive relationships in foreign direct investment. *European Economic Review*, *51*(5), 1303–1325. [https://doi.org/10.1016/j.euroecorev.2006.08.006](https://www.google.com/search?q=https://doi.org/10.1016/j.euroecorev.2006.08.006)
* **Core empirical application:** Estimates spatial interdependence in U.S. outbound foreign direct investment (FDI) across European host nations. By integrating a spatial lag of neighboring inward FDI into a gravity model, the authors distinguish between horizontal export-platform FDI and pure vertical fragmentation.


* **Ertur, C., & Koch, W. (2007).** Growth, technological interdependence and spatial externalities: Theory and evidence. *Journal of Applied Econometrics*, *22*(6), 1033–1062. [https://doi.org/10.1002/jae.963](https://www.google.com/search?q=https://doi.org/10.1002/jae.963)
* **Core empirical application:** Extends the neoclassical Solow–Swan growth model to accommodate spatial technological externalities. Using cross-country output and capital investment data, they use an endogenous spatial lag specification to prove that physical capital investments generate transnational productivity spillovers.



---

### 4. Labor Economics & Spatial General Equilibrium

* **Redding, S. J., & Rossi-Hansberg, E. (2017).** Quantitative spatial economics. *Annual Review of Economics*, *9*(1), 21–58. [https://doi.org/10.1146/annurev-economics-063016-103713](https://www.google.com/search?q=https://doi.org/10.1146/annurev-economics-063016-103713)
* **Core empirical application:** Reviews the modern frontier uniting spatial econometrics with quantitative general equilibrium models. It details how bilateral commuter flows, localized housing supply elasticities, and congestion forces are structurally estimated to measure the general equilibrium incidence of place-based infrastructure and local labor demand shocks.


* **Moretti, E. (2011).** Local labor markets. In D. Card & O. Ashenfelter (Eds.), *Handbook of Labor Economics* (Vol. 4, Part B, pp. 1237–1313). Elsevier. [https://doi.org/10.1016/S0169-7218(11)02410-X](https://www.google.com/search?q=https://doi.org/10.1016/S0169-7218(11)02410-X)
* **Core empirical application:** Analyzes spatial equilibrium across cities, providing the empirical framework for estimating local multiplier effects (how adding one skilled tradable job generates non-tradable service jobs within and across surrounding commuting zones).



### 1. Spatial Econometrics in Real Estate & Hedonic Price Modeling

Real estate pricing was one of the earliest proving grounds for large-scale spatial autoregression because housing assets exhibit strong spatial autocorrelation due to shared neighborhood amenities, school boundaries, and regional labor markets.

* **Computational Sparse Autoregressions:**
Estimating spatial autoregressive models on tens of thousands of individual parcel transactions was historically bottlenecked by calculating log-determinants of large $N \times N$ weight matrices. Kelley Pace and Barry (1997) introduced sparse spatial autoregressions using Delaunay triangulations and sparse matrix algorithms. Using California block-group housing data ($N = 20{,}640$), they showed how simultaneous autoregressions capture local price spillovers and eliminate residual autocorrelation in hedonic pricing functions.

---

### 2. Point Pattern Modeling & Disease Cluster Detection in Public Health

In spatial epidemiology and public health, case coordinates (such as cancer diagnoses or respiratory illnesses) require point process models to differentiate true localized risk clusters from underlying population density variations.

* **Point Processes near Environmental Hazard Sources:**
Diggle (1990) established the foundational inhomogeneous Poisson point process framework to test whether the incidence of a rare disease is elevated near a suspected pollution source (such as an industrial incinerator or chemical waste dump). Rather than assuming a uniform background, the model conditions case intensity on the spatial density of non-diseased controls, testing whether risk decays monotonically with distance from the hazard point.
* **Spatial Scan Statistics for Cluster Identification:**
Standard point pattern methods often suffer from multiple testing issues when scanning across a map without an *a priori* hypothesized cluster center. Han et al. (2019) extended the traditional combinatorial spatial scan statistic framework to a continuous kernel-based formulation, enabling detection of localized disease outbreaks with smooth spatial decay around epicenters.

---

### References

Diggle, P. J. (1990). A Point Process Modelling Approach to Raised Incidence of a Rare Phenomenon in the Vicinity of a Prespecified Point. *Journal of the Royal Statistical Society. Series A (Statistics in Society)*, *153*(3), 349. [https://doi.org/10.2307/2982977](https://doi.org/10.2307/2982977)
Cited by: 394

Han, M., Matheny, M., & Phillips, J. M. (2019). The Kernel Spatial Scan Statistic. *Proceedings of the 27th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems*, 349–358. [https://doi.org/10.1145/3347146.3359101](https://doi.org/10.1145/3347146.3359101)
Cited by: 15

Kelley Pace, R., & Barry, R. (1997). Sparse spatial autoregressions. *Statistics & Probability Letters*, *33*(3), 291–297. [https://doi.org/10.1016/s0167-7152(96)00140-x](https://doi.org/10.1016/s0167-7152(96)00140-x)
Cited by: 1182



In transportation safety, civil engineering, and actuarial science, car accidents and auto insurance claims are fundamentally spatial phenomena. Standard Generalized Linear Models (GLMs) assume claims or crashes are independent across space, but neighboring intersections and rating territories share road geometry, commuter volume, policing intensity, and weather.

Researchers approach this using three distinct spatial statistical frameworks:

---

### 1. Actuarial Science: Spatial Smoothing & Territorial Ratemaking (Areal / Lattice)

Auto insurers divide geographic regions into rating territories (e.g., ZIP codes, Forward Sortation Areas, census tracts) to determine territorial base premiums.

* **The Problem:** Sparsely populated rural territories suffer from low credibility (high sample variance in claim counts), while sharp price discontinuities at administrative borders cause unfair discrimination and policyholder churn.
* **The Method:** Bayesian spatial smoothing models—specifically Besag–York–Mollié (BYM) or Conditional Autoregressive (CAR) models. These models decompose territory risk into an unstructured random effect (local noise) and a structured spatial random effect (borrowing credibility from contiguous neighboring postal codes).
* **Representative Study:**
* **Xu & Zhou (2025)** — *Spatial Modeling of Auto Insurance Loss Metrics to Uncover Impact of COVID-19 on Territorial Risk* (*Mathematics*).
* **Application:** Used Bayesian spatial CAR models and Local Moran’s $I$ across postal rating units (FSAs) to model spatial dependencies in auto insurance loss frequency, loss cost, and loss severity. They showed that incorporating spatial autocorrelation across neighboring rating units eliminates artificial boundary discontinuities and captures regional shifts in driving risk.



---

### 2. Traffic Safety Analysis: Point Processes Constrained to Linear Networks

Standard spatial point process models (e.g., 2D planar Poisson processes or planar kernel density estimation) measure distance using Euclidean metric ($L_2$). In car accidents, Euclidean distance is mathematically invalid: crashes cannot occur in fields, parks, or rivers between streets; they are strictly constrained to a 1D graph network.

* **The Problem:** Planar Kernel Density Estimation (KDE) produces "leakage"—it estimates high crash risk on buildings or open land between two parallel avenues.
* **The Method:** **Network-Constrained Point Processes (Network KDE & Net-Voronoi)**. Distance $d_G(x, y)$ is calculated as shortest-path distance along the directed graph of the street network. Intensity $\lambda(s)$ is modeled as an inhomogeneous Poisson process along 1D line segments.
* **Representative Studies:**
* **Okabe, Satoh, & Sugihara (2009)** — *A kernel density estimation method for networks, its computational method and a GIS-based tool* (*International Journal of Geographical Information Science*). Formulated equal-split and continuous Kernel Density Estimation constrained to street graphs, resolving the planar bias when detecting high-collision blackspots.
* **Baddeley, Nair, Rakshit, & McSwiggan (2017)** — *Point process models on linear networks* (*Statistica Neerlandica*). Provided the foundational likelihood theory and Poisson regression formulations for spatial point pattern analysis constrained to linear network graphs ($L$).



---

### 3. Spatial Econometrics: Macro-Level Crash Frequency & Spatial Spillover

Transportation planners frequently aggregate collision data to census blocks, Traffic Analysis Zones (TAZs), or county jurisdictions to determine where to allocate state highway funds.

* **The Problem:** Treating adjacent TAZs as independent observations ignores spatial spillover: an un-signaled highway intersection in Zone $A$ causes traffic jams and rear-end collisions that spill back into upstream Zone $B$.
* **The Method:** Spatial Autoregressive Models (SAR, SEM, and Spatial Durbin) combined with Poisson or Negative Binomial regression (Spatial Count Data Models) using road-connectivity weight matrices $W$.
* **Representative Studies:**
* **Quddus (2008)** — *Modelling area-wide count of road traffic accidents using spatial statistical methods* (*Accident Analysis & Prevention*). One of the most-cited papers applying CAR priors and spatial autoregressive specifications to area-wide crash frequencies in London, proving that standard negative binomial regressions severely underestimate parameter standard errors by ignoring spatial correlation.
* **Aguero-Valverde & Jovanis (2006)** — *Spatial analysis of fatal and injury crashes in Pennsylvania* (*Accident Analysis & Prevention*). Applied hierarchical Bayesian spatial error modeling with intrinsic conditional autoregression (ICAR) to crash rates across Pennsylvania counties, demonstrating how regional deprivation and weather effects propagate spatially.



---

### Summary of Spatial Methods Across the Two Fields

| Application Domain | Method / Model | Spatial Structure | What it Solves / Uncovers |
| --- | --- | --- | --- |
| **Auto Insurance Territory Ratemaking** | Bayesian CAR / BYM models | Areal / Lattice (ZIP codes, FSAs) | Borrows statistical credibility from neighboring territories to price low-volume rural areas smoothly. |
| **Crash "Blackspot" Detection** | Network-constrained Inhomogeneous Poisson / Net-KDE | Point patterns on 1D linear graphs | Eliminates Euclidean "leakage" into non-road spaces and correctly computes distance via the street network. |
| **Regional Highway Safety Planning** | Spatial Durbin Count / SAR / SEM models | Areal / Lattice (Traffic Analysis Zones) | Captures congestion and collision spillover across administrative and highway boundaries. |

---

### References

* **Aguero-Valverde, J., & Jovanis, P. P. (2006).** Spatial analysis of fatal and injury crashes in Pennsylvania. *Accident Analysis & Prevention*, *38*(3), 618–625. [https://doi.org/10.1016/j.aap.2005.12.006](https://www.google.com/search?q=https://doi.org/10.1016/j.aap.2005.12.006)
* **Baddeley, A., Nair, G., Rakshit, S., & McSwiggan, G. (2017).** Point process models on linear networks. *Statistica Neerlandica*, *71*(4), 304–336. [https://doi.org/10.1111/stan.12117](https://www.google.com/search?q=https://doi.org/10.1111/stan.12117)
* **Okabe, A., Satoh, T., & Sugihara, K. (2009).** A kernel density estimation method for networks, its computational method and a GIS-based tool. *International Journal of Geographical Information Science*, *23*(1), 7–32. [https://doi.org/10.1080/13658810802475491](https://www.google.com/search?q=https://doi.org/10.1080/13658810802475491)
* **Quddus, M. A. (2008).** Modelling area-wide count of road traffic accidents using spatial statistical methods. *Accident Analysis & Prevention*, *40*(4), 1458–1466. [https://doi.org/10.1016/j.aap.2008.03.007](https://www.google.com/search?q=https://doi.org/10.1016/j.aap.2008.03.007)
* **Xu, S., & Zhou, J. (2025).** Spatial Modeling of Auto Insurance Loss Metrics to Uncover Impact of COVID-19 on Territorial Risk. *Mathematics*, *13*(9), 1416. [https://doi.org/10.3390/math13091416](https://www.google.com/search?q=https://doi.org/10.3390/math13091416)
