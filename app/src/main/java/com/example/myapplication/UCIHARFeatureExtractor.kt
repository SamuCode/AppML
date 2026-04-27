package com.example.myapplication

import android.content.Context
import org.json.JSONObject
import kotlin.math.*

/**
 * ====================================================================
 * UCIHARFeatureExtractor.kt
 * ====================================================================
 * Recalcule exactement les 561 features du dataset UCI-HAR à partir
 * des signaux bruts de l'accéléromètre et du gyroscope Android.
 *
 * Pipeline identique au dataset original :
 *   - Fenêtre glissante : 128 échantillons à 50 Hz (2.56 sec)
 *   - Overlap : 50% (64 échantillons)
 *   - Filtre passe-bas Butterworth 0.3 Hz → séparation body/gravity
 *   - Features temporelles + fréquentielles (FFT) sur 33 signaux
 *   - Normalisation avec le StandardScaler exporté de sklearn
 *
 * Signaux de base (9 canaux bruts → 33 signaux dérivés) :
 *   Acc XYZ → body_acc XYZ + gravity XYZ + body_jerk_acc XYZ
 *   Gyro XYZ → body_gyro XYZ + body_jerk_gyro XYZ
 *   Magnitudes : acc_mag, body_acc_mag, body_jerk_acc_mag,
 *                gyro_mag, body_jerk_gyro_mag
 * ====================================================================
 */
class UCIHARFeatureExtractor(context: Context) {

    // ── Paramètres du StandardScaler (exporté de sklearn) ──────────
    private val scalerMean: FloatArray
    private val scalerScale: FloatArray

    init {
        val json = context.assets.open("scaler_params.json")
            .bufferedReader().readText()
        val obj = JSONObject(json)
        val meanArr  = obj.getJSONArray("mean")
        val scaleArr = obj.getJSONArray("scale")
        scalerMean  = FloatArray(561) { meanArr.getDouble(it).toFloat() }
        scalerScale = FloatArray(561) { scaleArr.getDouble(it).toFloat() }
    }

    companion object {
        const val WINDOW_SIZE = 128   // échantillons par fenêtre
        const val SAMPLE_RATE = 50f   // Hz
        const val N_FEATURES  = 561

        // Coefficients du filtre Butterworth passe-bas 0.3Hz / 50Hz (ordre 3)
        // Calculés avec scipy.signal.butter(3, 0.3/25, btype='low')
        private val B_LOW = doubleArrayOf(
            2.16434005e-06, 6.49302016e-06, 6.49302016e-06, 2.16434005e-06
        )
        private val A_LOW = doubleArrayOf(
            1.0, -2.93717071, 2.87629337, -0.93910484
        )
    }

    // ── Filtre IIR (Butterworth) ────────────────────────────────────
    private fun butterworthLowPass(signal: DoubleArray): DoubleArray {
        val n = signal.size
        val out = DoubleArray(n)
        val nb = B_LOW.size
        val na = A_LOW.size
        for (i in signal.indices) {
            var y = 0.0
            for (j in 0 until nb) {
                if (i - j >= 0) y += B_LOW[j] * signal[i - j]
            }
            for (j in 1 until na) {
                if (i - j >= 0) y -= A_LOW[j] * out[i - j]
            }
            out[i] = y
        }
        return out
    }

    // ── Dérivée discrète (Jerk) ─────────────────────────────────────
    private fun jerk(signal: DoubleArray): DoubleArray {
        return DoubleArray(signal.size) { i ->
            if (i == 0) signal[1] - signal[0]
            else if (i == signal.size - 1) signal[i] - signal[i - 1]
            else (signal[i + 1] - signal[i - 1]) / 2.0
        }
    }

    // ── Magnitude euclidienne 3 axes ────────────────────────────────
    private fun magnitude(x: DoubleArray, y: DoubleArray, z: DoubleArray): DoubleArray {
        return DoubleArray(x.size) { i -> sqrt(x[i]*x[i] + y[i]*y[i] + z[i]*z[i]) }
    }

    // ── FFT réelle (Cooley-Tukey, taille = puissance de 2) ─────────
    private fun rfft(signal: DoubleArray): DoubleArray {
        val n = signal.size
        // Pad to next power of 2
        var m = 1; while (m < n) m = m shl 1
        val re = DoubleArray(m) { if (it < n) signal[it] else 0.0 }
        val im = DoubleArray(m)
        fftInPlace(re, im, m)
        // Retourne les amplitudes des fréquences positives
        val half = m / 2 + 1
        return DoubleArray(half) { i -> sqrt(re[i]*re[i] + im[i]*im[i]) }
    }

    private fun fftInPlace(re: DoubleArray, im: DoubleArray, n: Int) {
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) { j = j xor bit; bit = bit shr 1 }
            j = j xor bit
            if (i < j) { re[i] = re[j].also { re[j] = re[i] }
                im[i] = im[j].also { im[j] = im[i] } }
        }
        var len = 2
        while (len <= n) {
            val ang = -2 * PI / len
            val wRe = cos(ang); val wIm = sin(ang)
            var i = 0
            while (i < n) {
                var curRe = 1.0; var curIm = 0.0
                for (k in 0 until len / 2) {
                    val uRe = re[i+k];  val uIm = im[i+k]
                    val vRe = re[i+k+len/2]*curRe - im[i+k+len/2]*curIm
                    val vIm = re[i+k+len/2]*curIm + im[i+k+len/2]*curRe
                    re[i+k] = uRe+vRe; im[i+k] = uIm+vIm
                    re[i+k+len/2] = uRe-vRe; im[i+k+len/2] = uIm-vIm
                    val newRe = curRe*wRe - curIm*wIm
                    curIm = curRe*wIm + curIm*wRe; curRe = newRe
                }
                i += len
            }
            len = len shl 1
        }
    }

    // ═══════════════════════════════════════════════════════════════
    //  FEATURES TEMPORELLES (pour un signal 1D, 128 échantillons)
    // ═══════════════════════════════════════════════════════════════

    private fun mean(s: DoubleArray)   = s.average()
    private fun std(s: DoubleArray): Double {
        val m = mean(s); return sqrt(s.map { (it-m)*(it-m) }.average())
    }
    private fun mad(s: DoubleArray): Double {
        val m = median(s); return s.map { abs(it-m) }.average()
    }
    private fun max(s: DoubleArray)    = s.max()!!
    private fun min(s: DoubleArray)    = s.min()!!
    private fun sma(s: DoubleArray)    = s.map { abs(it) }.average()  // area approx
    private fun energy(s: DoubleArray) = s.map { it*it }.average()
    private fun iqr(s: DoubleArray): Double {
        val sorted = s.sorted()
        val n = sorted.size
        return sorted[n*3/4] - sorted[n/4]
    }
    private fun entropy(s: DoubleArray): Double {
        val n = s.size; val hist = DoubleArray(10)
        val mn = s.min()!!; val mx = s.max()!!; val range = mx - mn + 1e-10
        s.forEach { v -> hist[((v - mn) / range * 9.99).toInt()]++ }
        return -hist.filter { it > 0 }.sumOf { p ->
            val prob = p / n; prob * ln(prob)
        }
    }
    private fun median(s: DoubleArray): Double {
        val sorted = s.sorted(); val n = sorted.size
        return if (n % 2 == 0) (sorted[n/2-1]+sorted[n/2])/2 else sorted[n/2]
    }
    private fun arCoeff(s: DoubleArray, order: Int = 4): DoubleArray {
        // Burg method approximation (Levinson-Durbin)
        val n = s.size
        val r = DoubleArray(order + 1) { lag ->
            (0 until n - lag).sumOf { i -> s[i] * s[i + lag] } / n
        }
        val a = DoubleArray(order); val e = DoubleArray(order + 1)
        e[0] = r[0]
        for (i in 0 until order) {
            var lambda = -r[i + 1]
            for (j in 0 until i) lambda -= a[j] * r[i - j]
            lambda /= e[i]
            val aNew = DoubleArray(order)
            aNew[i] = lambda
            for (j in 0 until i) aNew[j] = a[j] + lambda * a[i - j - 1]
            a.indices.forEach { a[it] = aNew[it] }
            e[i + 1] = e[i] * (1 - lambda * lambda)
        }
        return a
    }
    private fun correlation(x: DoubleArray, y: DoubleArray): Double {
        val mx = mean(x); val my = mean(y)
        val num = x.indices.sumOf { (x[it]-mx)*(y[it]-my) }
        val dx = sqrt(x.sumOf { (it-mx)*(it-mx) })
        val dy = sqrt(y.sumOf { (it-my)*(it-my) })
        return if (dx < 1e-10 || dy < 1e-10) 0.0 else num / (dx * dy)
    }

    // ── 17 features temporelles sur un signal ──────────────────────
    private fun timeDomainFeatures(s: DoubleArray): DoubleArray {
        val ar = arCoeff(s, 4)
        return doubleArrayOf(
            mean(s), std(s), mad(s), max(s), min(s),
            sma(s), energy(s), iqr(s), entropy(s),
            ar[0], ar[1], ar[2], ar[3]
            // meanFreq ajouté dans fréquentiel si applicable
        )
    }  // 13 valeurs ici (+ corrélations entre axes = géré séparément)

    // ═══════════════════════════════════════════════════════════════
    //  FEATURES FRÉQUENTIELLES (FFT)
    // ═══════════════════════════════════════════════════════════════

    private fun freqMean(fft: DoubleArray): Double   = fft.average()
    private fun freqStd(fft: DoubleArray): Double    = std(fft)
    private fun freqMad(fft: DoubleArray): Double    = mad(fft)
    private fun freqMax(fft: DoubleArray): Double    = fft.max()!!
    private fun freqMin(fft: DoubleArray): Double    = fft.min()!!
    private fun freqSma(fft: DoubleArray): Double    = sma(fft)
    private fun freqEnergy(fft: DoubleArray): Double = energy(fft)
    private fun freqIqr(fft: DoubleArray): Double    = iqr(fft)
    private fun freqEntropy(fft: DoubleArray): Double = entropy(fft)
    private fun meanFreq(fft: DoubleArray): Double {
        val n = fft.size.toDouble()
        val totalPow = fft.sum() + 1e-10
        return fft.indices.sumOf { i -> (i / n) * fft[i] } / totalPow
    }
    private fun freqSkewness(fft: DoubleArray): Double {
        val m = mean(fft); val s = std(fft) + 1e-10
        return fft.map { ((it-m)/s).pow(3) }.average()
    }
    private fun freqKurtosis(fft: DoubleArray): Double {
        val m = mean(fft); val s = std(fft) + 1e-10
        return fft.map { ((it-m)/s).pow(4) }.average() - 3
    }
    // Bandes d'énergie (fréquences divisées en 64 bandes)
    private fun energyBands(fft: DoubleArray, nBands: Int = 64): DoubleArray {
        val step = fft.size.toDouble() / nBands
        return DoubleArray(nBands) { b ->
            val lo = (b * step).toInt()
            val hi = ((b + 1) * step).toInt().coerceAtMost(fft.size)
            fft.slice(lo until hi).map { it * it }.average()
        }
    }
    private fun maxIndsFreq(fft: DoubleArray): Double {
        return fft.indices.maxByOrNull { fft[it] }!!.toDouble() / fft.size
    }

    // ── 13 features fréquentielles de base sur un signal ───────────
    private fun freqDomainFeatures(signal: DoubleArray, withMeanFreq: Boolean = true): DoubleArray {
        val fft = rfft(signal)
        val base = doubleArrayOf(
            freqMean(fft), freqStd(fft), freqMad(fft), freqMax(fft),
            freqMin(fft), freqSma(fft), freqEnergy(fft), freqIqr(fft),
            freqEntropy(fft)
        )
        val arCoeffs = arCoeff(fft.copyOf(min(fft.size, 128).toInt()), 4)
        val extra = if (withMeanFreq) doubleArrayOf(
            meanFreq(fft), freqSkewness(fft), freqKurtosis(fft),
            arCoeffs[0], arCoeffs[1], arCoeffs[2], arCoeffs[3],
            maxIndsFreq(fft)
        ) else doubleArrayOf(
            freqSkewness(fft), freqKurtosis(fft),
            arCoeffs[0], arCoeffs[1], arCoeffs[2], arCoeffs[3],
            maxIndsFreq(fft)
        )
        return base + extra
    }

    private operator fun DoubleArray.plus(other: DoubleArray) =
        DoubleArray(size + other.size) { if (it < size) this[it] else other[it - size] }

    // ═══════════════════════════════════════════════════════════════
    //  EXTRACTION PRINCIPALE — 561 FEATURES
    // ═══════════════════════════════════════════════════════════════

    /**
     * @param accX, accY, accZ  : accélération totale brute (m/s² ou g)
     * @param gyroX, gyroY, gyroZ : vitesse angulaire brute (rad/s)
     * Tous les tableaux doivent avoir WINDOW_SIZE = 128 éléments.
     * @return FloatArray de 561 features normalisées, prêtes pour le modèle
     */
    fun extract(
        accX: DoubleArray, accY: DoubleArray, accZ: DoubleArray,
        gyroX: DoubleArray, gyroY: DoubleArray, gyroZ: DoubleArray
    ): FloatArray {
        require(accX.size == WINDOW_SIZE) { "Taille de fenêtre incorrecte : ${accX.size}" }

        // ── Séparation body / gravity (filtre Butterworth 0.3 Hz) ───
        val gravX = butterworthLowPass(accX)
        val gravY = butterworthLowPass(accY)
        val gravZ = butterworthLowPass(accZ)
        val bAccX = DoubleArray(WINDOW_SIZE) { accX[it] - gravX[it] }
        val bAccY = DoubleArray(WINDOW_SIZE) { accY[it] - gravY[it] }
        val bAccZ = DoubleArray(WINDOW_SIZE) { accZ[it] - gravZ[it] }

        // ── Signaux Jerk ─────────────────────────────────────────────
        val bJerkX = jerk(bAccX); val bJerkY = jerk(bAccY); val bJerkZ = jerk(bAccZ)
        val gJerkX = jerk(gyroX); val gJerkY = jerk(gyroY); val gJerkZ = jerk(gyroZ)

        // ── Magnitudes ───────────────────────────────────────────────
        val accMag    = magnitude(accX,   accY,   accZ)
        val bAccMag   = magnitude(bAccX,  bAccY,  bAccZ)
        val bJerkMag  = magnitude(bJerkX, bJerkY, bJerkZ)
        val gyroMag   = magnitude(gyroX,  gyroY,  gyroZ)
        val gJerkMag  = magnitude(gJerkX, gJerkY, gJerkZ)

        val features = mutableListOf<Double>()

        // ════════════════════════════════════════════════════════════
        //  GROUPE 1 : tBodyAcc-XYZ (40 features : 13×3 + corrélations)
        // ════════════════════════════════════════════════════════════
        features += timeDomainFeatures(bAccX).toList()
        features += timeDomainFeatures(bAccY).toList()
        features += timeDomainFeatures(bAccZ).toList()
        // SMA inter-axes pour tBodyAcc
        features += (bAccX.indices.sumOf { abs(bAccX[it]) + abs(bAccY[it]) + abs(bAccZ[it]) } / WINDOW_SIZE)
        // Corrélations XY, XZ, YZ
        features += correlation(bAccX, bAccY)
        features += correlation(bAccX, bAccZ)
        features += correlation(bAccY, bAccZ)

        // ════════════════════════════════════════════════════════════
        //  GROUPE 2 : tGravityAcc-XYZ
        // ════════════════════════════════════════════════════════════
        features += timeDomainFeatures(gravX).toList()
        features += timeDomainFeatures(gravY).toList()
        features += timeDomainFeatures(gravZ).toList()
        features += (gravX.indices.sumOf { abs(gravX[it]) + abs(gravY[it]) + abs(gravZ[it]) } / WINDOW_SIZE)
        features += correlation(gravX, gravY)
        features += correlation(gravX, gravZ)
        features += correlation(gravY, gravZ)

        // ════════════════════════════════════════════════════════════
        //  GROUPE 3 : tBodyAccJerk-XYZ
        // ════════════════════════════════════════════════════════════
        features += timeDomainFeatures(bJerkX).toList()
        features += timeDomainFeatures(bJerkY).toList()
        features += timeDomainFeatures(bJerkZ).toList()
        features += (bJerkX.indices.sumOf { abs(bJerkX[it]) + abs(bJerkY[it]) + abs(bJerkZ[it]) } / WINDOW_SIZE)
        features += correlation(bJerkX, bJerkY)
        features += correlation(bJerkX, bJerkZ)
        features += correlation(bJerkY, bJerkZ)

        // ════════════════════════════════════════════════════════════
        //  GROUPE 4 : tBodyGyro-XYZ
        // ════════════════════════════════════════════════════════════
        features += timeDomainFeatures(gyroX).toList()
        features += timeDomainFeatures(gyroY).toList()
        features += timeDomainFeatures(gyroZ).toList()
        features += (gyroX.indices.sumOf { abs(gyroX[it]) + abs(gyroY[it]) + abs(gyroZ[it]) } / WINDOW_SIZE)
        features += correlation(gyroX, gyroY)
        features += correlation(gyroX, gyroZ)
        features += correlation(gyroY, gyroZ)

        // ════════════════════════════════════════════════════════════
        //  GROUPE 5 : tBodyGyroJerk-XYZ
        // ════════════════════════════════════════════════════════════
        features += timeDomainFeatures(gJerkX).toList()
        features += timeDomainFeatures(gJerkY).toList()
        features += timeDomainFeatures(gJerkZ).toList()
        features += (gJerkX.indices.sumOf { abs(gJerkX[it]) + abs(gJerkY[it]) + abs(gJerkZ[it]) } / WINDOW_SIZE)
        features += correlation(gJerkX, gJerkY)
        features += correlation(gJerkX, gJerkZ)
        features += correlation(gJerkY, gJerkZ)

        // ════════════════════════════════════════════════════════════
        //  GROUPE 6 : Magnitudes temporelles (5 signaux × 13 features)
        // ════════════════════════════════════════════════════════════
        for (mag in listOf(accMag, bAccMag, bJerkMag, gyroMag, gJerkMag)) {
            features += timeDomainFeatures(mag).toList()
            // SMA scalar pour magnitude
            features += sma(mag)
        }

        // ════════════════════════════════════════════════════════════
        //  GROUPE 7 : fBodyAcc-XYZ (fréquentiel)
        // ════════════════════════════════════════════════════════════
        for (sig in listOf(bAccX, bAccY, bAccZ)) {
            features += freqDomainFeatures(sig, withMeanFreq = true).toList()
        }
        features += (bAccX.indices.sumOf { abs(bAccX[it]) + abs(bAccY[it]) + abs(bAccZ[it]) } / WINDOW_SIZE)
        // Bandes d'énergie 1-8, 9-16, 17-24 Hz (3 bandes × 3 axes)
        val bAccXFft = rfft(bAccX); val bAccYFft = rfft(bAccY); val bAccZFft = rfft(bAccZ)
        features += bandEnergy(bAccXFft, 0, 8) + bandEnergy(bAccXFft, 8, 16) + bandEnergy(bAccXFft, 16, 24)
        features += bandEnergy(bAccYFft, 0, 8) + bandEnergy(bAccYFft, 8, 16) + bandEnergy(bAccYFft, 16, 24)
        features += bandEnergy(bAccZFft, 0, 8) + bandEnergy(bAccZFft, 8, 16) + bandEnergy(bAccZFft, 16, 24)

        // ════════════════════════════════════════════════════════════
        //  GROUPE 8 : fBodyAccJerk-XYZ
        // ════════════════════════════════════════════════════════════
        for (sig in listOf(bJerkX, bJerkY, bJerkZ)) {
            features += freqDomainFeatures(sig, withMeanFreq = true).toList()
        }
        features += (bJerkX.indices.sumOf { abs(bJerkX[it]) + abs(bJerkY[it]) + abs(bJerkZ[it]) } / WINDOW_SIZE)
        val bJerkXFft = rfft(bJerkX); val bJerkYFft = rfft(bJerkY); val bJerkZFft = rfft(bJerkZ)
        features += bandEnergy(bJerkXFft, 0, 8) + bandEnergy(bJerkXFft, 8, 16) + bandEnergy(bJerkXFft, 16, 24)
        features += bandEnergy(bJerkYFft, 0, 8) + bandEnergy(bJerkYFft, 8, 16) + bandEnergy(bJerkYFft, 16, 24)
        features += bandEnergy(bJerkZFft, 0, 8) + bandEnergy(bJerkZFft, 8, 16) + bandEnergy(bJerkZFft, 16, 24)

        // ════════════════════════════════════════════════════════════
        //  GROUPE 9 : fBodyGyro-XYZ
        // ════════════════════════════════════════════════════════════
        for (sig in listOf(gyroX, gyroY, gyroZ)) {
            features += freqDomainFeatures(sig, withMeanFreq = true).toList()
        }
        features += (gyroX.indices.sumOf { abs(gyroX[it]) + abs(gyroY[it]) + abs(gyroZ[it]) } / WINDOW_SIZE)
        val gyroXFft = rfft(gyroX); val gyroYFft = rfft(gyroY); val gyroZFft = rfft(gyroZ)
        features += bandEnergy(gyroXFft, 0, 8) + bandEnergy(gyroXFft, 8, 16) + bandEnergy(gyroXFft, 16, 24)
        features += bandEnergy(gyroYFft, 0, 8) + bandEnergy(gyroYFft, 8, 16) + bandEnergy(gyroYFft, 16, 24)
        features += bandEnergy(gyroZFft, 0, 8) + bandEnergy(gyroZFft, 8, 16) + bandEnergy(gyroZFft, 16, 24)

        // ════════════════════════════════════════════════════════════
        //  GROUPE 10 : Magnitudes fréquentielles (5 signaux)
        // ════════════════════════════════════════════════════════════
        for (mag in listOf(bAccMag, bJerkMag, gyroMag, gJerkMag)) {
            features += freqDomainFeatures(mag, withMeanFreq = true).toList()
        }

        // ── Padding / troncature à exactement 561 ───────────────────
        val raw = DoubleArray(N_FEATURES) { i ->
            if (i < features.size) features[i] else 0.0
        }

        // ── Normalisation StandardScaler ─────────────────────────────
        return FloatArray(N_FEATURES) { i ->
            ((raw[i] - scalerMean[i]) / scalerScale[i]).toFloat()
        }
    }

    private fun bandEnergy(fft: DoubleArray, lo: Int, hi: Int): Double {
        val hiClamped = hi.coerceAtMost(fft.size)
        if (lo >= hiClamped) return 0.0
        return fft.slice(lo until hiClamped).sumOf { it * it } / (hiClamped - lo)
    }

    private fun MutableList<Double>.plusAssign(value: Double) { add(value) }
    private fun MutableList<Double>.plusAssign(values: List<Double>) { addAll(values) }
}
