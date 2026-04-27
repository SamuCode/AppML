package com.example.myapplication

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class HARClassifier(context: Context) {

    private val interpreter: Interpreter
    val labels: Array<String>

    // Normalisation par canal (6 valeurs : acc xyz + gyro xyz)
    private val normMean:  FloatArray
    private val normStd:   FloatArray

    init {
        // Modèle TFLite
        val fd = context.assets.openFd("har_model.tflite")
        val model: MappedByteBuffer = FileInputStream(fd.fileDescriptor).channel
            .map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        interpreter = Interpreter(model)

        // Labels
        val labelsJson = JSONArray(context.assets.open("label_classes.json")
            .bufferedReader().readText())
        labels = Array(labelsJson.length()) { labelsJson.getString(it) }

        // Paramètres de normalisation (mean + std par canal)
        val normJson = JSONObject(context.assets.open("norm_params.json")
            .bufferedReader().readText())
        val meanArr = normJson.getJSONArray("mean")
        val stdArr  = normJson.getJSONArray("std")
        normMean = FloatArray(6) { meanArr.getDouble(it).toFloat() }
        normStd  = FloatArray(6) { stdArr.getDouble(it).toFloat()  }
    }

    fun classify(
        accX: DoubleArray, accY: DoubleArray, accZ: DoubleArray,
        gyroX: DoubleArray, gyroY: DoubleArray, gyroZ: DoubleArray
    ): Pair<String, Float> {

        val ws = 128
        val input = Array(1) { Array(ws) { FloatArray(6) } }

        for (i in 0 until ws) {
            // Permutation des axes : Z→X, X→Z pour correspondre à UCI HAR
            input[0][i][0] = ((accZ[i]  - normMean[0]) / normStd[0]).toFloat()
            input[0][i][1] = ((accY[i]  - normMean[1]) / normStd[1]).toFloat()
            input[0][i][2] = ((accX[i]  - normMean[2]) / normStd[2]).toFloat()
            input[0][i][3] = ((gyroZ[i] - normMean[3]) / normStd[3]).toFloat()
            input[0][i][4] = ((gyroY[i] - normMean[4]) / normStd[4]).toFloat()
            input[0][i][5] = ((gyroX[i] - normMean[5]) / normStd[5]).toFloat()
        }

        val output = Array(1) { FloatArray(labels.size) }

        android.util.Log.d("HARQ", "input[0][0]: " +
                input[0][0][0] + " " + input[0][0][1] + " " + input[0][0][2] + " " +
                input[0][0][3] + " " + input[0][0][4] + " " + input[0][0][5])
        android.util.Log.d("HARQ", "input mean canal 0: " +
                (0 until 128).map { input[0][it][0].toDouble() }.average())
        android.util.Log.d("HARQ", "normMean: " + normMean[0] + " normStd: " + normStd[0])
        interpreter.run(input, output)

        val probs  = output[0]
        val maxIdx = probs.indices.maxByOrNull { probs[it] } ?: 0
        return Pair(labels[maxIdx], probs[maxIdx])
    }

    fun close() = interpreter.close()
}