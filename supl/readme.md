# 1. Predictions

In `supl/predictions` we include the RGB & NIR predictions for five random samples from the IID set, together with the corresponding ground truth.

The table below shows the performance of the model reported in the paper (Table 1) in comparison to the baselines reported in [EarthNet2021](https://arxiv.org/pdf/2104.10066.pdf).

<table style="border-collapse: collapse; border: medium none; border-spacing: 0px;">
	<caption>
		Comparison of our model againts the baseline models on the IID and OOD test sets. For our model we report the average scores over five runs with different random initializations.
	</caption>
	<tr>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="border-bottom: 0px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;" colspan="6">
			<b>IID</b>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="border-bottom: 0px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;" colspan="6">
			<b>OOD</b>
		</td>
	</tr>
	<tr>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>ENS</b>
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b></b>
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>MAD</b>
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>OLS</b>
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>EMD</b>
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>SSIM</b>
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>ENS</b>
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>MAD</b>
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>OLS</b>
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>EMD</b>
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>SSIM</b>
		</td>
	</tr>
	<tr>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			Persistance (baseline-1)
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2625
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2315
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3239
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2099
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3265
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2587
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2248
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3236
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2123
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3112
		</td>
	</tr>
	<tr>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			Channel-U-Net (baseline-2)
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2902
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2482
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3381
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2336
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3973
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2854
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2402
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3390
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2371
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3721
		</td>
	</tr>
	<tr>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			Arcon (baseline-3)
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2803
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2414
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3216
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2258
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3863
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2655
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2314
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3088
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2177
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3432
		</td>
	</tr>
	<tr>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			ConvLSTM
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>0.3266</b>
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>0.2638</b>
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>0.3513</b>
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>0.2623</b>
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>0.5565</b>
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>0.3204</b>
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>0.2541</b>
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>0.3522</b>
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>0.2660</b>
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>0.5125</b>
		</td>
	</tr>
</table>


# 2. Simulations
In the paper, we only showed a sample obtained by perturbing the amount of rainfall, for a limited number of timesteps (Figure 3). 
Here we additionally include five more samples, also for the other two meteorological conditions, i.e. sea level pressure and temperature, and with a larger number of timesteps. The predictions, shown as animations, can be found in the corresponding subdirectories (i.e. `supl/predictions/rainfall`, `supl/predictions/sea_level_pressure`, `supl/predictions/temperature`). 

For each of the three categories, we show below the average evaluation scores to investigate if the best performance if achieved when the actual scenario (i.e. no perturbation) is used.
   
## 2.1. Rainfall

The results below correspond to those reported in the paper in Table 3.

<table style="border-collapse: collapse; border: medium none; border-spacing: 0px;">
	<caption>
		Influence of five artificially generated rainfall scenarios on the evaluation scores, using a single model. The first column shows the average difference (over the entire dataset) between the original values and the perturbed ones. The row in bold corresponds to the actual scenario.
	</caption>
	<tr>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;" rowspan="2">
			<br>
			<br>
			 average daily
			<br>
			 rainfall change (mm)
		</td>
		<td style="border-bottom: 0px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;" colspan="6">
			<b>IID</b>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="border-bottom: 0px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;" colspan="6">
			<b>OOD</b>
		</td>
	</tr>
	<tr>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>ENS</b>
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b></b>
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>MAD</b>
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>OLS</b>
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>EMD</b>
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>SSIM</b>
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>ENS</b>
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>MAD</b>
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>OLS</b>
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>EMD</b>
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>SSIM</b>
		</td>
	</tr>
	<tr>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			-0.8
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3187
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2591
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3449
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2564
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.5286
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3130
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2493
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3479
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2610
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.4848
		</td>
	</tr>
	<tr>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			-0.4
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3244
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2624
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3498
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2606
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.5482
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3181
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2523
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3519
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2646
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.5025
		</td>
	</tr>
	<tr>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			+0.0
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>0.3262</b>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>0.2637</b>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>0.3512</b>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>0.2617</b>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>0.5547</b>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>0.3203</b>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>0.2539</b>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>0.3530</b>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>0.2659</b>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>0.5110</b>
		</td>
	</tr>
	<tr>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			+1.0
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3163
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2596
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3404
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2522
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.5294
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3054
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2476
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3364
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2517
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.4727
		</td>
	</tr>
	<tr>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			+2.0
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3062
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2558
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3307
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2422
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.5001
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2896
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2433
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3183
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2344
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.4363
		</td>
	</tr>
	<tr>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			+3.0
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2988
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2528
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3247
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2353
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.4764
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2807
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2408
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3087
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2252
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.4133
		</td>
	</tr>
</table>

## 2.2. Sea Level Pressure

<table style="border-collapse: collapse; border: medium none; border-spacing: 0px;">
	<caption>
		Influence of artificially generated sea level pressure scenarios on the evaluation scores, using a single model. The first column shows the average difference (over the entire dataset) between the original values and the perturbed ones. The row in bold corresponds to the actual scenario.
	</caption>
	<tr>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;" rowspan="2">
			<br>
			<br>
			 average daily sea level
			<br>
			 pressure change (mbar)
		</td>
		<td style="border-bottom: 0px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;" colspan="6">
			<b>IID</b>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="border-bottom: 0px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;" colspan="6">
			<b>OOD</b>
		</td>
	</tr>
	<tr>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>ENS</b>
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b></b>
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>MAD</b>
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>OLS</b>
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>EMD</b>
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>SSIM</b>
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>ENS</b>
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>MAD</b>
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>OLS</b>
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>EMD</b>
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>SSIM</b>
		</td>
	</tr>
	<tr>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			-15.0
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3100
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2534
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3416
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2490
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.4965
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3034
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2435
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3442
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2526
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.4520
		</td>
	</tr>
	<tr>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			-10.0
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3174
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2580
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3460
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2544
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.5245
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3110
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2476
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3487
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2588
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.4782
		</td>
	</tr>
	<tr>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			-5.0
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3236
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2620
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3498
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2593
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.5465
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3176
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2517
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3523
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2640
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.5012
		</td>
	</tr>
	<tr>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			+0.0
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>0.3262</b>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>0.2637</b>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>0.3512</b>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>0.2617</b>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>0.5547</b>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>0.3203</b>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>0.2539</b>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>0.3530</b>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>0.2659</b>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>0.5110</b>
		</td>
	</tr>
	<tr>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			+5.0
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3232
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2618
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3481
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2599
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.5443
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3174
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2529
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3502
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2633
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.5012
		</td>
	</tr>
	<tr>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			+10.0
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3145
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2569
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3410
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2536
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.5133
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3091
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2487
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3436
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2571
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.4714
		</td>
	</tr>
	<tr>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			+15.0
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3026
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2505
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3324
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2448
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.4688
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2983
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2432
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3358
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2494
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.4329
		</td>
	</tr>
</table>

## 2.3. Temperature
<table style="border-collapse: collapse; border: medium none; border-spacing: 0px;">
	<caption>
		Influence of artificially generated temperature scenarios on the evaluation scores, using a single model. All three temperature values (i.e. min, max and mean) were perturbed by the same amount. The first column shows the average difference (over the entire dataset) between the original values and the perturbed ones. The row in bold corresponds to the actual scenario.
	</caption>
	<tr>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;" rowspan="2">
			<br>
			<br>
			 average daily
			<br>
			 temperature change (&#8451)
		</td>
		<td style="border-bottom: 0px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;" colspan="6">
			<b>IID</b>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="border-bottom: 0px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;" colspan="6">
			<b>OOD</b>
		</td>
	</tr>
	<tr>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>ENS</b>
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b></b>
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>MAD</b>
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>OLS</b>
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>EMD</b>
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>SSIM</b>
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>ENS</b>
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>MAD</b>
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>OLS</b>
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>EMD</b>
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>SSIM</b>
		</td>
	</tr>
	<tr>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			-6.0
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2970
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2482
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3241
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2390
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.4611
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2932
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2405
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3265
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2451
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.4268
		</td>
	</tr>
	<tr>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			-4.0
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3126
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2561
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3373
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2518
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.5120
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3085
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2481
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3410
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2570
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.4730
		</td>
	</tr>
	<tr>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			-2.0
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3228
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2616
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3472
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2595
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.5447
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3179
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2530
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3502
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2640
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.5031
		</td>
	</tr>
	<tr>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			+0.0
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>0.3262</b>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>0.2637</b>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>0.3512</b>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>0.2617</b>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>0.5547</b>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>0.3203</b>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>0.2539</b>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>0.3530</b>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>0.2659</b>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>0.5110</b>
		</td>
	</tr>
	<tr>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			+2.0
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3237
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2623
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3497
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2595
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.5464
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3172
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2515
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3522
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2637
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.4994
		</td>
	</tr>
	<tr>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			+4.0
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3173
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2585
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3453
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2544
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.5236
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3102
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2471
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3486
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2585
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.4741
		</td>
	</tr>
	<tr>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			+6.0
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3095
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2538
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3405
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2487
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.4933
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3021
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2426
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.3439
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.2524
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			0.4444
		</td>
	</tr>
</table>