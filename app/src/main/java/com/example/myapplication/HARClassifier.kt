package com.example.myapplication

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONArray
import java.nio.FloatBuffer

class HARClassifier(context: Context) {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    val labels: Array<String>

    init {
        val modelBytes = context.assets.open("har_model.onnx").readBytes()
        session = env.createSession(modelBytes)

        val labelsJson = JSONArray(context.assets.open("label_classes.json")
            .bufferedReader().readText())
        labels = Array(labelsJson.length()) { labelsJson.getString(it) }
    }

    fun classify(features: FloatArray): Pair<String, Float> {
        val inputName = session.inputNames.iterator().next()
        val shape     = longArrayOf(1, features.size.toLong())
        val tensor    = OnnxTensor.createTensor(env, FloatBuffer.wrap(features), shape)

        val output = session.run(mapOf(inputName to tensor))
        val raw    = output[0].value

        val probs: FloatArray = when (raw) {
            is Array<*>   -> (raw as Array<FloatArray>)[0]
            is FloatArray -> raw
            else          -> FloatArray(labels.size)
        }

        tensor.close()
        output.close()

        val maxIdx = probs.indices.maxByOrNull { probs[it] } ?: 0
        return Pair(labels[maxIdx], probs[maxIdx])
    }

    fun close() {
        session.close()
        env.close()
    }
}