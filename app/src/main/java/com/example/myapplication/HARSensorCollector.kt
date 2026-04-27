package com.example.myapplication

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import java.util.concurrent.ConcurrentLinkedDeque

class HARSensorCollector(
    context: Context,
    private val onWindowReady: (
        accX: DoubleArray, accY: DoubleArray, accZ: DoubleArray,
        gyroX: DoubleArray, gyroY: DoubleArray, gyroZ: DoubleArray
    ) -> Unit
) : SensorEventListener {

    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager

    private val accBuf  = Array(3) { ConcurrentLinkedDeque<Double>() }
    private val gyroBuf = Array(3) { ConcurrentLinkedDeque<Double>() }

    private var lastAccTs  = 0L
    private var lastGyroTs = 0L
    private val targetIntervalNs = 1_000_000_000L / 50

    private var samplesProcessed = 0

    fun start() {
        val acc  = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        val gyro = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        sensorManager.registerListener(this, acc,  SensorManager.SENSOR_DELAY_GAME)
        sensorManager.registerListener(this, gyro, SensorManager.SENSOR_DELAY_GAME)
    }

    fun stop() = sensorManager.unregisterListener(this)

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                if (event.timestamp - lastAccTs >= targetIntervalNs * 0.8) {
                    for (i in 0..2) {
                        // Convertir m/s² → g
                        accBuf[i].addLast(event.values[i].toDouble() / 9.80665)
                        if (accBuf[i].size > 256) accBuf[i].pollFirst()
                    }
                    lastAccTs = event.timestamp
                    checkWindowReady()
                }
            }
            Sensor.TYPE_GYROSCOPE -> {
                if (event.timestamp - lastGyroTs >= targetIntervalNs * 0.8) {
                    for (i in 0..2) {
                        gyroBuf[i].addLast(event.values[i].toDouble())
                        if (gyroBuf[i].size > 256) gyroBuf[i].pollFirst()
                    }
                    lastGyroTs = event.timestamp
                }
            }
        }
    }

    private fun checkWindowReady() {
        val ws = 128
        if (accBuf[0].size >= ws && gyroBuf[0].size >= ws) {
            samplesProcessed++
            if (samplesProcessed % 64 == 0) {
                val accLists  = Array(3) { i -> accBuf[i].toList().takeLast(ws) }
                val gyroLists = Array(3) { i -> gyroBuf[i].toList().takeLast(ws) }
                if (accLists.all { it.size == ws } && gyroLists.all { it.size == ws }) {
                    onWindowReady(
                        accLists[0].toDoubleArray(), accLists[1].toDoubleArray(), accLists[2].toDoubleArray(),
                        gyroLists[0].toDoubleArray(), gyroLists[1].toDoubleArray(), gyroLists[2].toDoubleArray()
                    )
                }
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
}