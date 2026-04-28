package com.example.myapplication

import android.content.Context
import org.json.JSONObject
import kotlin.math.*

/**
 * ====================================================================
 * UCIHARFeatureExtractor.kt  — VERSION CORRIGÉE
 * ====================================================================
 * Suit EXACTEMENT l'ordre des 561 colonnes du dataset UCI-HAR :
 *
 *  0–39    tBodyAcc XYZ         (40 features)
 *  40–79   tGravityAcc XYZ      (40 features)
 *  80–119  tBodyAccJerk XYZ     (40 features)
 * 120–159  tBodyGyro XYZ        (40 features)
 * 160–199  tBodyGyroJerk XYZ    (40 features)
 * 200–212  tBodyAccMag          (13 features)
 * 213–225  tGravityAccMag       (13 features)
 * 226–238  tBodyAccJerkMag      (13 features)
 * 239–251  tBodyGyroMag         (13 features)
 * 252–264  tBodyGyroJerkMag     (13 features)
 * 265–343  fBodyAcc XYZ         (79 features)
 * 344–422  fBodyAccJerk XYZ     (79 features)
 * 423–501  fBodyGyro XYZ        (79 features)
 * 502–514  fBodyAccMag          (13 features)
 * 515–527  fBodyBodyAccJerkMag  (13 features)
 * 528–540  fBodyBodyGyroMag     (13 features)
 * 541–553  fBodyBodyGyroJerkMag (13 features)
 * 554–560  angle()              (7 features)
 * ====================================================================
 */
class UCIHARFeatureExtractor(context: Context) {

    private val scalerMean: FloatArray
    private val scalerScale: FloatArray

    init {
        val json = context.assets.open("scaler_params.json").bufferedReader().readText()
        val obj  = JSONObject(json)
        val m    = obj.getJSONArray("mean")
        val s    = obj.getJSONArray("scale")
        scalerMean  = FloatArray(N_FEATURES) { m.getDouble(it).toFloat() }
        scalerScale = FloatArray(N_FEATURES) { s.getDouble(it).toFloat() }
    }

    companion object {
        const val WINDOW_SIZE = 128
        const val SAMPLE_RATE = 50f
        const val N_FEATURES  = 561

        // Butterworth passe-bas ordre 3, fc=0.3Hz, fs=50Hz
        private val B = doubleArrayOf(2.16434005e-06, 6.49302016e-06, 6.49302016e-06, 2.16434005e-06)
        private val A = doubleArrayOf(1.0, -2.93717071, 2.87629337, -0.93910484)

        // Bandes d'énergie UCI-HAR (index dans le vecteur FFT de taille 64)
        // FFT sur 128 points → 64 bins positifs, chaque bin = 50/128 ≈ 0.39 Hz
        // Les bandes du dataset sont en numéros de bins 1-based → on convertit en 0-based
        val BANDS = listOf(
            0 to 7,   // 1,8
            8 to 15,  // 9,16
            16 to 23, // 17,24
            24 to 31, // 25,32
            32 to 39, // 33,40
            40 to 47, // 41,48
            48 to 55, // 49,56
            56 to 63, // 57,64
            0 to 15,  // 1,16
            16 to 31, // 17,32
            32 to 47, // 33,48
            48 to 63, // 49,64
            0 to 23,  // 1,24
            24 to 47  // 25,48
        )
    }

    // ── Filtre Butterworth IIR ──────────────────────────────────────
    private fun lowPass(x: DoubleArray): DoubleArray {
        val y = DoubleArray(x.size)
        for (i in x.indices) {
            y[i] = B[0]*x[i]
            if (i >= 1) y[i] += B[1]*x[i-1] - A[1]*y[i-1]
            if (i >= 2) y[i] += B[2]*x[i-2] - A[2]*y[i-2]
            if (i >= 3) y[i] += B[3]*x[i-3] - A[3]*y[i-3]
        }
        return y
    }

    // ── Dérivée discrète ────────────────────────────────────────────
    private fun jerk(x: DoubleArray) = DoubleArray(x.size) { i ->
        when (i) {
            0           -> x[1] - x[0]
            x.size - 1  -> x[i] - x[i-1]
            else        -> (x[i+1] - x[i-1]) / 2.0
        }
    }

    // ── Magnitude ───────────────────────────────────────────────────
    private fun mag(x: DoubleArray, y: DoubleArray, z: DoubleArray) =
        DoubleArray(x.size) { i -> sqrt(x[i]*x[i] + y[i]*y[i] + z[i]*z[i]) }

    // ── Statistiques de base ────────────────────────────────────────
    private fun mean(s: DoubleArray) = s.average()
    private fun std(s: DoubleArray): Double {
        val m = mean(s); return sqrt(s.sumOf { (it-m)*(it-m) } / s.size)
    }
    private fun mad(s: DoubleArray): Double {
        val m = median(s); return s.map { abs(it-m) }.average()
    }
    private fun median(s: DoubleArray): Double {
        val sorted = s.sorted(); val n = sorted.size
        return if (n % 2 == 0) (sorted[n/2-1]+sorted[n/2])/2.0 else sorted[n/2]
    }
    private fun iqr(s: DoubleArray): Double {
        val sorted = s.sorted(); val n = sorted.size
        return sorted[(n*3/4).coerceAtMost(n-1)] - sorted[n/4]
    }
    private fun energy(s: DoubleArray) = s.sumOf { it*it } / s.size
    private fun entropy(s: DoubleArray): Double {
        val mn = s.min()!!; val mx = s.max()!!; val range = mx - mn + 1e-10
        val hist = DoubleArray(10)
        s.forEach { v -> hist[((v-mn)/range*9.99).toInt().coerceIn(0,9)]++ }
        return -hist.filter { it > 0 }.sumOf { p -> val pr = p/s.size; pr * ln(pr) }
    }
    private fun sma3(x: DoubleArray, y: DoubleArray, z: DoubleArray) =
        (x.sumOf { abs(it) } + y.sumOf { abs(it) } + z.sumOf { abs(it) }) / x.size
    private fun correlation(x: DoubleArray, y: DoubleArray): Double {
        val mx = mean(x); val my = mean(y)
        val num = x.indices.sumOf { (x[it]-mx)*(y[it]-my) }
        val dx  = sqrt(x.sumOf { (it-mx)*(it-mx) })
        val dy  = sqrt(y.sumOf { (it-my)*(it-my) })
        return if (dx < 1e-10 || dy < 1e-10) 0.0 else num/(dx*dy)
    }
    private fun arCoeff(s: DoubleArray, order: Int = 4): DoubleArray {
        val n = s.size
        val r = DoubleArray(order+1) { lag -> (0 until n-lag).sumOf { s[it]*s[it+lag] } / n }
        val a = DoubleArray(order); val e = DoubleArray(order+1); e[0] = r[0]
        for (i in 0 until order) {
            var lam = -r[i+1]; for (j in 0 until i) lam -= a[j]*r[i-j]
            lam /= e[i].coerceAtLeast(1e-10)
            val aN = DoubleArray(order); aN[i] = lam
            for (j in 0 until i) aN[j] = a[j] + lam*a[i-j-1]
            aN.copyInto(a); e[i+1] = e[i]*(1 - lam*lam)
        }
        return a
    }

    // ── FFT réelle ──────────────────────────────────────────────────
    private fun rfft(x: DoubleArray): DoubleArray {
        var m = 1; while (m < x.size) m = m shl 1
        val re = DoubleArray(m) { if (it < x.size) x[it] else 0.0 }
        val im = DoubleArray(m)
        // Bit-reversal
        var j = 0
        for (i in 1 until m) {
            var bit = m shr 1
            while (j and bit != 0) { j = j xor bit; bit = bit shr 1 }
            j = j xor bit
            if (i < j) { re[i] = re[j].also { re[j] = re[i] }; im[i] = im[j].also { im[j] = im[i] } }
        }
        // Butterfly
        var len = 2
        while (len <= m) {
            val ang = -2*PI/len; val wRe = cos(ang); val wIm = sin(ang)
            var ii = 0
            while (ii < m) {
                var cRe = 1.0; var cIm = 0.0
                for (k in 0 until len/2) {
                    val uR = re[ii+k]; val uI = im[ii+k]
                    val vR = re[ii+k+len/2]*cRe - im[ii+k+len/2]*cIm
                    val vI = re[ii+k+len/2]*cIm + im[ii+k+len/2]*cRe
                    re[ii+k] = uR+vR; im[ii+k] = uI+vI
                    re[ii+k+len/2] = uR-vR; im[ii+k+len/2] = uI-vI
                    val nR = cRe*wRe - cIm*wIm; cIm = cRe*wIm + cIm*wRe; cRe = nR
                }
                ii += len
            }
            len = len shl 1
        }
        return DoubleArray(m/2) { i -> sqrt(re[i]*re[i] + im[i]*im[i]) }
    }

    // ── Features fréquentielles sur spectre ─────────────────────────
    private fun meanFreq(fft: DoubleArray): Double {
        val total = fft.sum() + 1e-10
        return fft.indices.sumOf { i -> (i.toDouble()/fft.size) * fft[i] } / total
    }
    private fun skewness(s: DoubleArray): Double {
        val m = mean(s); val sd = std(s) + 1e-10
        return s.sumOf { ((it-m)/sd).pow(3) } / s.size
    }
    private fun kurtosis(s: DoubleArray): Double {
        val m = mean(s); val sd = std(s) + 1e-10
        return s.sumOf { ((it-m)/sd).pow(4) } / s.size - 3
    }
    private fun maxInds(fft: DoubleArray): Double =
        -(fft.indices.maxByOrNull { fft[it] }!!.toDouble() + 1) / fft.size  // négatif comme UCI

    private fun bandEnergy(fft: DoubleArray, lo: Int, hi: Int): Double {
        val hiC = (hi+1).coerceAtMost(fft.size)
        if (lo >= hiC) return 0.0
        return fft.slice(lo until hiC).sumOf { it*it } / (hiC - lo)
    }

    // ════════════════════════════════════════════════════════════════
    //  13 FEATURES TEMPORELLES (pour un signal 1D)
    //  ordre : mean std mad max min sma energy iqr entropy ar1 ar2 ar3 ar4
    // ════════════════════════════════════════════════════════════════
    private fun timeFeatures13(s: DoubleArray): DoubleArray {
        val ar = arCoeff(s)
        return doubleArrayOf(
            mean(s), std(s), mad(s), s.max()!!, s.min()!!,
            s.sumOf { abs(it) } / s.size,   // sma scalaire (pour magnitudes)
            energy(s), iqr(s), entropy(s),
            ar[0], ar[1], ar[2], ar[3]
        )
    }

    // ════════════════════════════════════════════════════════════════
    //  40 FEATURES TEMPORELLES POUR UN GROUPE XYZ
    //  ordre UCI : mean×3 std×3 mad×3 max×3 min×3 sma iqr×3 entropy×3
    //              arCoeff×12 correlation×3
    // ════════════════════════════════════════════════════════════════
    private fun timeXYZ(
        x: DoubleArray, y: DoubleArray, z: DoubleArray,
        out: MutableList<Double>
    ) {
        // mean XYZ
        out += mean(x); out += mean(y); out += mean(z)
        // std XYZ
        out += std(x);  out += std(y);  out += std(z)
        // mad XYZ
        out += mad(x);  out += mad(y);  out += mad(z)
        // max XYZ
        out += x.max()!!; out += y.max()!!; out += z.max()!!
        // min XYZ
        out += x.min()!!; out += y.min()!!; out += z.min()!!
        // sma (scalaire inter-axes)
        out += sma3(x, y, z)
        // energy XYZ
        out += energy(x); out += energy(y); out += energy(z)
        // iqr XYZ
        out += iqr(x); out += iqr(y); out += iqr(z)
        // entropy XYZ
        out += entropy(x); out += entropy(y); out += entropy(z)
        // arCoeff X(1-4) Y(1-4) Z(1-4)
        val arX = arCoeff(x); val arY = arCoeff(y); val arZ = arCoeff(z)
        arX.forEach { out += it }; arY.forEach { out += it }; arZ.forEach { out += it }
        // correlation XY XZ YZ
        out += correlation(x, y); out += correlation(x, z); out += correlation(y, z)
    }

    // ════════════════════════════════════════════════════════════════
    //  13 FEATURES TEMPORELLES POUR UNE MAGNITUDE
    //  ordre : mean std mad max min sma energy iqr entropy ar1 ar2 ar3 ar4
    // ════════════════════════════════════════════════════════════════
    private fun timeMag(s: DoubleArray, out: MutableList<Double>) {
        out += mean(s); out += std(s); out += mad(s)
        out += s.max()!!; out += s.min()!!
        out += s.sumOf { abs(it) } / s.size   // sma
        out += energy(s); out += iqr(s); out += entropy(s)
        val ar = arCoeff(s); ar.forEach { out += it }
    }

    // ════════════════════════════════════════════════════════════════
    //  79 FEATURES FRÉQUENTIELLES POUR UN GROUPE XYZ
    //  ordre UCI par axe (26 features) + sma + bandsEnergy×14×3 axes
    //
    //  Par axe (26) : mean std mad max min sma energy iqr entropy
    //                 maxInds meanFreq skewness kurtosis
    //                 bandsEnergy×14 (dans cet ordre dans le dataset)  ← NON
    //
    //  En réalité l'ordre UCI est :
    //  mean×3 std×3 mad×3 max×3 min×3 sma energy×3 iqr×3 entropy×3
    //  maxInds×3 meanFreq×3 skewness×3 kurtosis×3
    //  bandsEnergy×14 (axe X) bandsEnergy×14 (axe Y) bandsEnergy×14 (axe Z)
    // ════════════════════════════════════════════════════════════════
    private fun freqXYZ(
        x: DoubleArray, y: DoubleArray, z: DoubleArray,
        out: MutableList<Double>
    ) {
        val fX = rfft(x); val fY = rfft(y); val fZ = rfft(z)

        out += mean(fX); out += mean(fY); out += mean(fZ)
        out += std(fX);  out += std(fY);  out += std(fZ)
        out += mad(fX);  out += mad(fY);  out += mad(fZ)
        out += fX.max()!!; out += fY.max()!!; out += fZ.max()!!
        out += fX.min()!!; out += fY.min()!!; out += fZ.min()!!
        // sma fréquentiel inter-axes
        out += (fX.sumOf { abs(it) } + fY.sumOf { abs(it) } + fZ.sumOf { abs(it) }) / fX.size
        out += energy(fX); out += energy(fY); out += energy(fZ)
        out += iqr(fX); out += iqr(fY); out += iqr(fZ)
        out += entropy(fX); out += entropy(fY); out += entropy(fZ)
        out += maxInds(fX); out += maxInds(fY); out += maxInds(fZ)
        out += meanFreq(fX); out += meanFreq(fY); out += meanFreq(fZ)
        out += skewness(fX); out += skewness(fY); out += skewness(fZ)
        out += kurtosis(fX); out += kurtosis(fY); out += kurtosis(fZ)
        // bandsEnergy : 14 bandes × 3 axes (X puis Y puis Z)
        for (fft in listOf(fX, fY, fZ)) {
            BANDS.forEach { (lo, hi) -> out += bandEnergy(fft, lo, hi) }
        }
    }

    // ════════════════════════════════════════════════════════════════
    //  13 FEATURES FRÉQUENTIELLES POUR UNE MAGNITUDE
    //  ordre : mean std mad max min sma energy iqr entropy maxInds meanFreq skewness kurtosis
    // ════════════════════════════════════════════════════════════════
    private fun freqMag(s: DoubleArray, out: MutableList<Double>) {
        val fft = rfft(s)
        out += mean(fft); out += std(fft); out += mad(fft)
        out += fft.max()!!; out += fft.min()!!
        out += fft.sumOf { abs(it) } / fft.size   // sma
        out += energy(fft); out += iqr(fft); out += entropy(fft)
        out += maxInds(fft); out += meanFreq(fft)
        out += skewness(fft); out += kurtosis(fft)
    }

    // ════════════════════════════════════════════════════════════════
    //  7 FEATURES ANGLE
    //  angle(tBodyAccMean, gravity)
    //  angle(tBodyAccJerkMean, gravityMean)
    //  angle(tBodyGyroMean, gravityMean)
    //  angle(tBodyGyroJerkMean, gravityMean)
    //  angle(X, gravityMean)
    //  angle(Y, gravityMean)
    //  angle(Z, gravityMean)
    // ════════════════════════════════════════════════════════════════
    private fun angleBetween(ax: Double, ay: Double, az: Double,
                             bx: Double, by: Double, bz: Double): Double {
        val dot  = ax*bx + ay*by + az*bz
        val magA = sqrt(ax*ax + ay*ay + az*az) + 1e-10
        val magB = sqrt(bx*bx + by*by + bz*bz) + 1e-10
        return dot / (magA * magB)   // UCI retourne cos(angle), pas l'angle
    }

    // ════════════════════════════════════════════════════════════════
    //  EXTRACTION PRINCIPALE
    // ════════════════════════════════════════════════════════════════
    fun extract(
        accX: DoubleArray, accY: DoubleArray, accZ: DoubleArray,
        gyroX: DoubleArray, gyroY: DoubleArray, gyroZ: DoubleArray
    ): FloatArray {
        require(accX.size == WINDOW_SIZE)

        // ── Signaux dérivés ─────────────────────────────────────────
        val gravX  = lowPass(accX);  val gravY = lowPass(accY);  val gravZ = lowPass(accZ)
        val bAccX  = DoubleArray(WINDOW_SIZE) { accX[it]-gravX[it] }
        val bAccY  = DoubleArray(WINDOW_SIZE) { accY[it]-gravY[it] }
        val bAccZ  = DoubleArray(WINDOW_SIZE) { accZ[it]-gravZ[it] }
        val bJerkX = jerk(bAccX);  val bJerkY = jerk(bAccY);  val bJerkZ = jerk(bAccZ)
        val gJerkX = jerk(gyroX);  val gJerkY = jerk(gyroY);  val gJerkZ = jerk(gyroZ)

        // ── Magnitudes ──────────────────────────────────────────────
        val tAccMag     = mag(accX,   accY,   accZ)
        val tGravMag    = mag(gravX,  gravY,  gravZ)
        val tBAccMag    = mag(bAccX,  bAccY,  bAccZ)
        val tBJerkMag   = mag(bJerkX, bJerkY, bJerkZ)
        val tGyroMag    = mag(gyroX,  gyroY,  gyroZ)
        val tGJerkMag   = mag(gJerkX, gJerkY, gJerkZ)

        val out = mutableListOf<Double>()

        // ── 0–39   : tBodyAcc XYZ ───────────────────────────────────
        timeXYZ(bAccX, bAccY, bAccZ, out)

        // ── 40–79  : tGravityAcc XYZ ────────────────────────────────
        timeXYZ(gravX, gravY, gravZ, out)

        // ── 80–119 : tBodyAccJerk XYZ ───────────────────────────────
        timeXYZ(bJerkX, bJerkY, bJerkZ, out)

        // ── 120–159: tBodyGyro XYZ ───────────────────────────────────
        timeXYZ(gyroX, gyroY, gyroZ, out)

        // ── 160–199: tBodyGyroJerk XYZ ──────────────────────────────
        timeXYZ(gJerkX, gJerkY, gJerkZ, out)

        // ── 200–212: tBodyAccMag ─────────────────────────────────────
        timeMag(tBAccMag, out)

        // ── 213–225: tGravityAccMag ──────────────────────────────────
        timeMag(tGravMag, out)

        // ── 226–238: tBodyAccJerkMag ─────────────────────────────────
        timeMag(tBJerkMag, out)

        // ── 239–251: tBodyGyroMag ────────────────────────────────────
        timeMag(tGyroMag, out)

        // ── 252–264: tBodyGyroJerkMag ────────────────────────────────
        timeMag(tGJerkMag, out)

        // ── 265–343: fBodyAcc XYZ (79 features) ─────────────────────
        freqXYZ(bAccX, bAccY, bAccZ, out)

        // ── 344–422: fBodyAccJerk XYZ ────────────────────────────────
        freqXYZ(bJerkX, bJerkY, bJerkZ, out)

        // ── 423–501: fBodyGyro XYZ ───────────────────────────────────
        freqXYZ(gyroX, gyroY, gyroZ, out)

        // ── 502–514: fBodyAccMag ─────────────────────────────────────
        freqMag(tBAccMag, out)

        // ── 515–527: fBodyBodyAccJerkMag ─────────────────────────────
        freqMag(tBJerkMag, out)

        // ── 528–540: fBodyBodyGyroMag ────────────────────────────────
        freqMag(tGyroMag, out)

        // ── 541–553: fBodyBodyGyroJerkMag ────────────────────────────
        freqMag(tGJerkMag, out)

        // ── 554–560: angle() ─────────────────────────────────────────
        // gravityMean = moyenne de la composante gravité
        val gMeanX = mean(gravX); val gMeanY = mean(gravY); val gMeanZ = mean(gravZ)
        // tBodyAccMean
        val baMeanX = mean(bAccX); val baMeanY = mean(bAccY); val baMeanZ = mean(bAccZ)
        // tBodyAccJerkMean
        val bjMeanX = mean(bJerkX); val bjMeanY = mean(bJerkY); val bjMeanZ = mean(bJerkZ)
        // tBodyGyroMean
        val gyMeanX = mean(gyroX); val gyMeanY = mean(gyroY); val gyMeanZ = mean(gyroZ)
        // tBodyGyroJerkMean
        val gjMeanX = mean(gJerkX); val gjMeanY = mean(gJerkY); val gjMeanZ = mean(gJerkZ)

        out += angleBetween(baMeanX, baMeanY, baMeanZ, gMeanX,  gMeanY,  gMeanZ)   // 554
        out += angleBetween(bjMeanX, bjMeanY, bjMeanZ, gMeanX,  gMeanY,  gMeanZ)   // 555
        out += angleBetween(gyMeanX, gyMeanY, gyMeanZ, gMeanX,  gMeanY,  gMeanZ)   // 556
        out += angleBetween(gjMeanX, gjMeanY, gjMeanZ, gMeanX,  gMeanY,  gMeanZ)   // 557
        out += angleBetween(1.0, 0.0, 0.0,             gMeanX,  gMeanY,  gMeanZ)   // 558 angle(X, gravityMean)
        out += angleBetween(0.0, 1.0, 0.0,             gMeanX,  gMeanY,  gMeanZ)   // 559 angle(Y, gravityMean)
        out += angleBetween(0.0, 0.0, 1.0,             gMeanX,  gMeanY,  gMeanZ)   // 560 angle(Z, gravityMean)

        // ── Vérification ────────────────────────────────────────────
        check(out.size == N_FEATURES) {
            "Erreur : ${out.size} features générées au lieu de $N_FEATURES"
        }

        // ── Normalisation StandardScaler ─────────────────────────────
        return FloatArray(N_FEATURES) { i ->
            ((out[i] - scalerMean[i]) / scalerScale[i]).toFloat()
        }
    }

    // Helpers
    private fun MutableList<Double>.plusAssign(v: Double) { add(v) }
}
