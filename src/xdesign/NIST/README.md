# NIST DATA README

## About
Data in this directory is pulled from the NIST Database using the [`xrayplots`
API](https://github.com/iwancornelius/xrayplots).

An index of every element and compound available through NIST is found in `NIST_index.json`. This file is included with the distribution because it contained misplaced punctuation when fetched from the `xrayplots` API. `download_NIST_index.py` will re-generate this file from the API if there are ever updates to the NIST Database.

Attenuation data for elements and compounds used by `XDesign` is downloaded on demand and stored as `json` files in this directory.

## Reference
Hubbell, J.H. and Seltzer, S.M. (2004), Tables of X-Ray Mass Attenuation
Coefficients and Mass Energy-Absorption Coefficients (version 1.4).
[Online] Available: http://physics.nist.gov/xaamdi [2017, May 11].
National Institute of Standards and Technology, Gaithersburg, MD.

## Table of Elements

Z	|	Key	|	Element	|	Z/A	|	I [eV]	|	Density [g/cm^3]
---     |	---	|	---	    |	---	|	---	|	-----
1	|	H	|	Hydrogen	|	0.99212	|	19.2	|	8.375E-05
2	|	He	|	Helium	|	0.49968	|	41.8	|	1.663E-04
3	|	Li	|	Lithium	|	0.43221	|	40.0	|	5.340E-01
4	|	Be	|	Beryllium	|	0.44384	|	63.7	|	1.848E+00
5	|	B	|	Boron	|	0.46245	|	76.0	|	2.370E+00
6	|	C	|	Carbon_Graphite	|	0.49954	|	78.0	|	1.700E+00
7	|	N	|	Nitrogen	|	0.49976	|	82.0	|	1.165E-03
8	|	O	|	Oxygen	|	0.50002	|	95.0	|	1.332E-03
9	|	F	|	Fluorine	|	0.47372	|	115.0	|	1.580E-03
10	|	Ne	|	Neon	|	0.49555	|	137.0	|	8.385E-04
11	|	Na	|	Sodium	|	0.47847	|	149.0	|	9.710E-01
12	|	Mg	|	Magnesium	|	0.49373	|	156.0	|	1.740E+00
13	|	Al	|	Aluminum	|	0.48181	|	166.0	|	2.699E+00
14	|	Si	|	Silicon	|	0.49848	|	173.0	|	2.330E+00
15	|	P	|	Phosphorus	|	0.48428	|	173.0	|	2.200E+00
16	|	S	|	Sulfur	|	0.49897	|	180.0	|	2.000E+00
17	|	Cl	|	Chlorine	|	0.47951	|	174.0	|	2.995E-03
18	|	Ar	|	Argon	|	0.45059	|	188.0	|	1.662E-03
19	|	K	|	Potassium	|	0.48595	|	190.0	|	8.620E-01
20	|	Ca	|	Calcium	|	0.49903	|	191.0	|	1.550E+00
21	|	Sc	|	Scandium	|	0.46712	|	216.0	|	2.989E+00
22	|	Ti	|	Titanium	|	0.45948	|	233.0	|	4.540E+00
23	|	V	|	Vanadium	|	0.45150	|	245.0	|	6.110E+00
24	|	Cr	|	Chromium	|	0.46157	|	257.0	|	7.180E+00
25	|	Mn	|	Manganese	|	0.45506	|	272.0	|	7.440E+00
26	|	Fe	|	Iron	|	0.46556	|	286.0	|	7.874E+00
27	|	Co	|	Cobalt	|	0.45815	|	297.0	|	8.900E+00
28	|	Ni	|	Nickel	|	0.47708	|	311.0	|	8.902E+00
29	|	Cu	|	Copper	|	0.45636	|	322.0	|	8.960E+00
30	|	Zn	|	Zinc	|	0.45879	|	330.0	|	7.133E+00
31	|	Ga	|	Gallium	|	0.44462	|	334.0	|	5.904E+00
32	|	Ge	|	Germanium	|	0.44071	|	350.0	|	5.323E+00
33	|	As	|	Arsenic	|	0.44046	|	347.0	|	5.730E+00
34	|	Se	|	Selenium	|	0.43060	|	348.0	|	4.500E+00
35	|	Br	|	Bromine	|	0.43803	|	343.0	|	7.072E-03
36	|	Kr	|	Krypton	|	0.42959	|	352.0	|	3.478E-03
37	|	Rb	|	Rubidium	|	0.43291	|	363.0	|	1.532E+00
38	|	Sr	|	Strontium	|	0.43369	|	366.0	|	2.540E+00
39	|	Y	|	Yttrium	|	0.43867	|	379.0	|	4.469E+00
40	|	Zr	|	Zirconium	|	0.43848	|	393.0	|	6.506E+00
41	|	Nb	|	Niobium	|	0.44130	|	417.0	|	8.570E+00
42	|	Mo	|	Molybdenum	|	0.43777	|	424.0	|	1.022E+01
43	|	Tc	|	Technetium	|	0.43919	|	428.0	|	1.150E+01
44	|	Ru	|	Ruthenium	|	0.43534	|	441.0	|	1.241E+01
45	|	Rh	|	Rhodium	|	0.43729	|	449.0	|	1.241E+01
46	|	Pd	|	Palladium	|	0.43225	|	470.0	|	1.202E+01
47	|	Ag	|	Silver	|	0.43572	|	470.0	|	1.050E+01
48	|	Cd	|	Cadmium	|	0.42700	|	469.0	|	8.650E+00
49	|	In	|	Indium	|	0.42676	|	488.0	|	7.310E+00
50	|	Sn	|	Tin	|	0.42120	|	488.0	|	7.310E+00
51	|	Sb	|	Antimony	|	0.41889	|	487.0	|	6.691E+00
52	|	Te	|	Tellurium	|	0.40752	|	485.0	|	6.240E+00
53	|	I	|	Iodine	|	0.41764	|	491.0	|	4.930E+00
54	|	Xe	|	Xenon	|	0.41130	|	482.0	|	5.485E-03
55	|	Cs	|	Cesium	|	0.41383	|	488.0	|	1.873E+00
56	|	Ba	|	Barium	|	0.40779	|	491.0	|	3.500E+00
57	|	La	|	Lanthanum	|	0.41035	|	501.0	|	6.154E+00
58	|	Ce	|	Cerium	|	0.41395	|	523.0	|	6.657E+00
59	|	Pr	|	Praseodymium	|	0.41871	|	535.0	|	6.710E+00
60	|	Nd	|	Neodymium	|	0.41597	|	546.0	|	6.900E+00
61	|	Pm	|	Promethium	|	0.42094	|	560.0	|	7.220E+00
62	|	Sm	|	Samarium	|	0.41234	|	574.0	|	7.460E+00
63	|	Eu	|	Europium	|	0.41457	|	580.0	|	5.243E+00
64	|	Gd	|	Gadolinium	|	0.40699	|	591.0	|	7.900E+00
65	|	Tb	|	Terbium	|	0.40900	|	614.0	|	8.229E+00
66	|	Dy	|	Dysprosium	|	0.40615	|	628.0	|	8.550E+00
67	|	Ho	|	Holmium	|	0.40623	|	650.0	|	8.795E+00
68	|	Er	|	Erbium	|	0.40655	|	658.0	|	9.066E+00
69	|	Tm	|	Thulium	|	0.40844	|	674.0	|	9.321E+00
70	|	Yb	|	Ytterbium	|	0.40453	|	684.0	|	6.730E+00
71	|	Lu	|	Lutetium	|	0.40579	|	694.0	|	9.840E+00
72	|	Hf	|	Hafnium	|	0.40338	|	705.0	|	1.331E+01
73	|	Ta	|	Tantalum	|	0.40343	|	718.0	|	1.665E+01
74	|	W	|	Tungsten	|	0.40250	|	727.0	|	1.930E+01
75	|	Re	|	Rhenium	|	0.40278	|	736.0	|	2.102E+01
76	|	Os	|	Osmium	|	0.39958	|	746.0	|	2.257E+01
77	|	Ir	|	Iridium	|	0.40058	|	757.0	|	2.242E+01
78	|	Pt	|	Platinum	|	0.39984	|	790.0	|	2.145E+01
79	|	Au	|	Gold	|	0.40108	|	790.0	|	1.932E+01
80	|	Hg	|	Mercury	|	0.39882	|	800.0	|	1.355E+01
81	|	Tl	|	Thallium	|	0.39631	|	810.0	|	1.172E+01
82	|	Pb	|	Lead	|	0.39575	|	823.0	|	1.135E+01
83	|	Bi	|	Bismuth	|	0.39717	|	823.0	|	9.747E+00
84	|	Po	|	Polonium	|	0.40195	|	830.0	|	9.320E+00
85	|	At	|	Astatine	|	0.40479	|	825.0	|	1.000E+01
86	|	Rn	|	Radon	|	0.38736	|	794.0	|	9.066E-03
87	|	Fr	|	Francium	|	0.39010	|	827.0	|	1.000E+01
88	|	Ra	|	Radium	|	0.38934	|	826.0	|	5.000E+00
89	|	Ac	|	Actinium	|	0.39202	|	841.0	|	1.007E+01
90	|	Th	|	Thorium	|	0.38787	|	847.0	|	1.172E+01
91	|	Pa	|	Protactinium	|	0.39388	|	878.0	|	1.537E+01
92	|	U	|	Uranium	|	0.38651	|	890.0	|	1.895E+01
