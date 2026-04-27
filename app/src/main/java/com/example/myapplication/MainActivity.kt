package com.example.myapplication

import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.*

class MainActivity : AppCompatActivity() {

    private lateinit var collector:  HARSensorCollector
    private lateinit var classifier: HARClassifier

    private lateinit var extractor:  UCIHARFeatureExtractor

    private lateinit var tvActivity:   TextView
    private lateinit var tvConfidence: TextView
    private lateinit var tvStatus:     TextView
    private lateinit var tvSensorData: TextView

    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        tvActivity   = findViewById(R.id.tv_activity)
        tvConfidence = findViewById(R.id.tv_confidence)
        tvStatus     = findViewById(R.id.tv_status)
        tvSensorData = findViewById(R.id.tv_sensor_data)

        extractor  = UCIHARFeatureExtractor(this)
        classifier = HARClassifier(this)

        tvStatus.text = "⏳ Calibration... (2.56 sec)"

        collector = HARSensorCollector(this) { accX, accY, accZ, gyroX, gyroY, gyroZ ->
            scope.launch {
                try {
                    // Extraction des 561 features
                    val features = extractor.extract(accX, accY, accZ, gyroX, gyroY, gyroZ)
                    android.util.Log.d("HAR", "Nb features: " + features.size)
                    android.util.Log.d("HAR", "min=" + features.min() + " max=" + features.max() + " mean=" + features.average())
                    // Inférence TFLite
                    val (label, confidence) = classifier.classify(features)
                    val pct = (confidence * 100).toInt()

                    val emoji = when (label) {
                        "WALKING"            -> "🚶"
                        "WALKING_UPSTAIRS"   -> "🏔️"
                        "WALKING_DOWNSTAIRS" -> "⬇️"
                        "SITTING"            -> "💺"
                        "STANDING"           -> "🧍"
                        "LAYING"             -> "🛌"
                        else                 -> "❓"
                    }

                    withContext(Dispatchers.Main) {
                        tvActivity.text   = "$emoji $label"
                        tvConfidence.text = "Confiance : $pct%"
                        tvStatus.text     = "✅ Analyse en cours"
                        tvSensorData.text = "Acc x=%.2f  y=%.2f  z=%.2f (g)".format(
                            accX.last(), accY.last(), accZ.last()
                        )
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        tvStatus.text = "⚠️ Erreur: ${e.message}"
                    }
                }
            }
        }

        collector.start()
    }

    override fun onDestroy() {
        super.onDestroy()
        collector.stop()
        classifier.close()
        scope.cancel()
    }
}