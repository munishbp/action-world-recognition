# CNN+ConvLSTM — RESULTS.md fill-in snippets
Copy each block into the matching section of `RESULTS.md`.

## Main Results table
Replace the `CNN+ConvLSTM` row with:

```
| CNN+ConvLSTM | CNN+RNN | Kenneth | 0.2701 | 0.5314 | 0.2472 | 18.3M | 18.3M |
```

## Training Efficiency table
Replace the `CNN+ConvLSTM` row with:

```
| CNN+ConvLSTM | 9.63 | 6.18 | 8 | 224 | 32 | 15 | A100 SXM4 |
```

## Easiest 10 classes for CNN+ConvLSTM
(Use these to populate the `ConvLSTM` column of the Easiest table.)

| Rank | Class Name | ConvLSTM accuracy |
|------|------------|------------------:|
| 1 | Turning the camera left while filming something | 0.8357 |
| 2 | Pushing something from right to left | 0.8051 |
| 3 | Turning the camera right while filming something | 0.7716 |
| 4 | Approaching something with your camera | 0.7500 |
| 5 | Turning the camera downwards while filming something | 0.7258 |
| 6 | Pushing something from left to right | 0.7240 |
| 7 | Turning the camera upwards while filming something | 0.6583 |
| 8 | Pulling something from right to left | 0.6400 |
| 9 | Moving something and something closer to each other | 0.6388 |
| 10 | Moving away from something with your camera | 0.6364 |

## Hardest 10 classes for CNN+ConvLSTM
(Use these to populate the `ConvLSTM` column of the Hardest table.)

| Rank | Class Name | ConvLSTM accuracy |
|------|------------|------------------:|
| 1 | Trying to pour something into something, but missing so it spills next to it | 0.0000 |
| 2 | Spilling something onto something | 0.0000 |
| 3 | Spilling something next to something | 0.0000 |
| 4 | Spilling something behind something | 0.0000 |
| 5 | Putting something onto a slanted surface but it doesn't glide down | 0.0000 |
| 6 | Pushing something onto something | 0.0000 |
| 7 | Pulling something onto something | 0.0000 |
| 8 | Pretending to spread air onto something | 0.0000 |
| 9 | Pretending or trying and failing to twist something | 0.0000 |
| 10 | Pouring something onto something | 0.0000 |

## Full per-class accuracy (for cross-model tables)
If Arthur is building the cross-model Easiest/Hardest tables, give him this CSV or the raw `results/CNNConvLSTM_results.json` `per_class_acc` field.

```csv
class_index,class_name,convlstm_acc
0,Approaching something with your camera,0.7500
1,Attaching something to something,0.0066
2,Bending something so that it deforms,0.0065
3,Bending something until it breaks,0.0388
4,Burying something in something,0.1026
5,Closing something,0.0790
6,Covering something with something,0.6187
7,Digging something out of something,0.1818
8,Dropping something behind something,0.2126
9,Dropping something in front of something,0.3245
10,Dropping something into something,0.1798
11,Dropping something next to something,0.3358
12,Dropping something onto something,0.0684
13,Failing to put something into something because something does not fit,0.0000
14,Folding something,0.2596
15,Hitting something with something,0.1447
16,Holding something,0.1168
17,Holding something behind something,0.0658
18,Holding something in front of something,0.2301
19,Holding something next to something,0.2038
20,Holding something over something,0.1693
21,Laying something on the table on its side not upright,0.1053
22,Letting something roll along a flat surface,0.2442
23,Letting something roll down a slanted surface,0.2340
24,Letting something roll up a slanted surface so it rolls back down,0.1136
25,Lifting a surface with something on it but not enough for it to slide down,0.1053
26,Lifting a surface with something on it until it starts sliding down,0.1750
27,Lifting something up completely without letting it drop down,0.2376
28,Lifting something up completely then letting it drop down,0.3909
29,Lifting something with something on it,0.3333
30,Lifting up one end of something without letting it drop down,0.5676
31,Lifting up one end of something then letting it drop down,0.4775
32,Moving away from something with your camera,0.6364
33,Moving part of something,0.0000
34,Moving something across a surface until it falls down,0.0843
35,Moving something across a surface without it falling down,0.0000
36,Moving something and something away from each other,0.6083
37,Moving something and something closer to each other,0.6388
38,Moving something and something so they collide with each other,0.0000
39,Moving something and something so they pass each other,0.4118
40,Moving something away from something,0.4317
41,Moving something away from the camera,0.3214
42,Moving something closer to something,0.3474
43,Moving something down,0.5563
44,Moving something towards the camera,0.3649
45,Moving something up,0.3900
46,Opening something,0.0602
47,Picking something up,0.2010
48,Piling something up,0.0621
49,Plugging something into something,0.2237
50,Plugging something into something but pulling it right out as you remove your hand,0.1055
51,Poking a hole into some substance,0.0000
52,Poking a hole into something soft,0.0000
53,Poking a stack of something so the stack collapses,0.0571
54,Poking a stack of something without the stack collapsing,0.0000
55,Poking something so it slightly moves,0.0733
56,Poking something so lightly that it doesn't or almost doesn't move,0.2260
57,Poking something so that it falls over,0.1321
58,Poking something so that it spins around,0.0000
59,Pouring something into something,0.2770
60,Pouring something into something until it overflows,0.0000
61,Pouring something onto something,0.0000
62,Pouring something out of something,0.0380
63,Pretending or failing to wipe something off of something,0.0280
64,Pretending or trying and failing to twist something,0.0000
65,Pretending to be tearing something that is not tearable,0.1981
66,Pretending to close something without actually closing it,0.0094
67,Pretending to open something without actually opening it,0.0348
68,Pretending to pick something up,0.2105
69,Pretending to poke something,0.0200
70,Pretending to pour something out of something but something is empty,0.0357
71,Pretending to put something behind something,0.1739
72,Pretending to put something into something,0.1323
73,Pretending to put something next to something,0.3077
74,Pretending to put something on a surface,0.6321
75,Pretending to put something onto something,0.0526
76,Pretending to put something underneath something,0.1667
77,Pretending to scoop something up with something,0.0385
78,Pretending to spread air onto something,0.0000
79,Pretending to sprinkle air onto something,0.1552
80,Pretending to squeeze something,0.0625
81,Pretending to take something from somewhere,0.0160
82,Pretending to take something out of something,0.0638
83,Pretending to throw something,0.1277
84,Pretending to turn something upside down,0.3333
85,Pulling something from behind of something,0.0476
86,Pulling something from left to right,0.6059
87,Pulling something from right to left,0.6400
88,Pulling something onto something,0.0000
89,Pulling something out of something,0.0120
90,Pulling two ends of something but nothing happens,0.2754
91,Pulling two ends of something so that it gets stretched,0.3167
92,Pulling two ends of something so that it separates into two pieces,0.0351
93,Pushing something from left to right,0.7240
94,Pushing something from right to left,0.8051
95,Pushing something off of something,0.0278
96,Pushing something onto something,0.0000
97,Pushing something so it spins,0.0331
98,Pushing something so that it almost falls off but doesn't,0.1786
99,Pushing something so that it falls off the table,0.3187
100,Pushing something so that it slightly moves,0.1774
101,Pushing something with something,0.1593
102,Putting number of something onto something,0.1035
103,Putting something and something on the table,0.5179
104,Putting something behind something,0.2047
105,Putting something in front of something,0.2815
106,Putting something into something,0.2569
107,Putting something next to something,0.4187
108,Putting something on a flat surface without letting it roll,0.0405
109,Putting something on a surface,0.5267
110,Putting something on the edge of something so it is not supported and falls down,0.0563
111,Putting something onto a slanted surface but it doesn't glide down,0.0000
112,Putting something onto something,0.2302
113,Putting something onto something else that cannot support it so it falls down,0.0556
114,Putting something similar to other things that are already on the table,0.2604
115,Putting something that can't roll onto a slanted surface so it slides down,0.0217
116,Putting something that can't roll onto a slanted surface so it stays where it is,0.1667
117,Putting something that cannot actually stand upright upright on the table so it falls on its side,0.1905
118,Putting something underneath something,0.0508
119,Putting something upright on the table,0.1287
120,Putting something something and something on the table,0.3701
121,Removing something revealing something behind,0.3750
122,Rolling something on a flat surface,0.1675
123,Scooping something up with something,0.1077
124,Showing a photo of something to the camera,0.3462
125,Showing something behind something,0.4587
126,Showing something next to something,0.2330
127,Showing something on top of something,0.2435
128,Showing something to the camera,0.0038
129,Showing that something is empty,0.2537
130,Showing that something is inside something,0.0929
131,Something being deflected from something,0.0959
132,Something colliding with something and both are being deflected,0.0690
133,Something colliding with something and both come to a halt,0.0175
134,Something falling like a feather or paper,0.2038
135,Something falling like a rock,0.3528
136,Spilling something behind something,0.0000
137,Spilling something next to something,0.0000
138,Spilling something onto something,0.0000
139,Spinning something so it continues spinning,0.2536
140,Spinning something that quickly stops spinning,0.3450
141,Spreading something onto something,0.0156
142,Sprinkling something onto something,0.0781
143,Squeezing something,0.1981
144,Stacking number of something,0.2675
145,Stuffing something into something,0.0578
146,Taking one of many similar things on the table,0.4417
147,Taking something from somewhere,0.0312
148,Taking something out of something,0.2343
149,Tearing something into two pieces,0.5679
150,Tearing something just a little bit,0.4212
151,Throwing something,0.2880
152,Throwing something against something,0.1633
153,Throwing something in the air and catching it,0.3578
154,Throwing something in the air and letting it fall,0.1732
155,Throwing something onto a surface,0.0244
156,Tilting something with something on it slightly so it doesn't fall down,0.0846
157,Tilting something with something on it until it falls off,0.1630
158,Tipping something over,0.0596
159,Tipping something with something in it over so something in it falls out,0.0182
160,Touching (without moving) part of something,0.1927
161,Trying but failing to attach something to something because it doesn't stick,0.0395
162,Trying to bend something unbendable so nothing happens,0.2897
163,Trying to pour something into something but missing so it spills next to it,0.0000
164,Turning something upside down,0.5524
165,Turning the camera downwards while filming something,0.7258
166,Turning the camera left while filming something,0.8357
167,Turning the camera right while filming something,0.7716
168,Turning the camera upwards while filming something,0.6583
169,Twisting (wringing) something wet until water comes out,0.0400
170,Twisting something,0.0784
171,Uncovering something,0.5481
172,Unfolding something,0.2140
173,Wiping something off of something,0.2993
```

## Summary stats
- Top-1 accuracy: **27.01%**
- Top-5 accuracy: **53.14%**
- F1 (weighted): **0.2472**
- Best class: **Turning the camera left while filming something** (83.6%)
- Worst class: **Trying to pour something into something, but missing so it spills next to it** (0.0%)
- Classes above random (>0.57%): **154/174**
