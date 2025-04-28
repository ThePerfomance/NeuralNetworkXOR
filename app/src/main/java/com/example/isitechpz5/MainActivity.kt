package com.example.isitechpz5

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import kotlin.math.exp

class MainActivity : AppCompatActivity() {

    // Класс нейронной сети
    class NeuralNetwork(
        private val inputSize: Int,
        private val hiddenSize: Int,
        private val outputSize: Int,
        private val learningRate: Double
    ) {
        // Веса и смещения (инициализированы в диапазоне [-1, 1])
        private val weightsInputHidden = Array(inputSize) {
            DoubleArray(hiddenSize) { Math.random() * 2 - 1 }
        }
        private val biasHidden = DoubleArray(hiddenSize) { Math.random() * 2 - 1 }
        private val weightsHiddenOutput = DoubleArray(hiddenSize) { Math.random() * 2 - 1 }
        private var biasOutput = Math.random() * 2 - 1

        // Сигмоида и её производная
        private fun sigmoid(x: Double) = 1 / (1 + exp(-x))
        private fun sigmoidDerivative(x: Double) = sigmoid(x) * (1 - sigmoid(x))

        // Прямой проход
        fun forward(inputs: DoubleArray): Double {
            require(inputs.size == inputSize) { "Неверный размер входных данных" }

            val hiddenOutputs = DoubleArray(hiddenSize) { i ->
                var sum = 0.0
                for (j in inputs.indices) {
                    sum += inputs[j] * weightsInputHidden[j][i]
                }
                sum += biasHidden[i]
                sigmoid(sum)
            }

            var outputSum = 0.0
            for (i in hiddenOutputs.indices) {
                outputSum += hiddenOutputs[i] * weightsHiddenOutput[i]
            }
            outputSum += biasOutput
            return sigmoid(outputSum)
        }

        // Обучение
        fun train(inputs: DoubleArray, target: Double) {
            require(inputs.size == inputSize) { "Неверный размер входных данных" }

            // Прямой проход
            val hiddenOutputs = DoubleArray(hiddenSize) { i ->
                var sum = 0.0
                for (j in inputs.indices) {
                    sum += inputs[j] * weightsInputHidden[j][i]
                }
                sum += biasHidden[i]
                sigmoid(sum)
            }

            var outputSum = 0.0
            for (i in hiddenOutputs.indices) {
                outputSum += hiddenOutputs[i] * weightsHiddenOutput[i]
            }
            outputSum += biasOutput
            val output = sigmoid(outputSum)

            // Обратное распространение ошибки
            val outputError = target - output
            val outputDelta = outputError * sigmoidDerivative(outputSum)

            // Обновление весов выходного слоя
            for (i in weightsHiddenOutput.indices) {
                weightsHiddenOutput[i] += learningRate * outputDelta * hiddenOutputs[i]
            }
            biasOutput += learningRate * outputDelta

            // Ошибка скрытого слоя
            val hiddenDeltas = DoubleArray(hiddenSize) { i ->
                val error = outputDelta * weightsHiddenOutput[i]
                error * sigmoidDerivative(hiddenOutputs[i])
            }

            // Обновление весов скрытого слоя
            for (i in inputs.indices) {
                for (j in hiddenDeltas.indices) {
                    weightsInputHidden[i][j] += learningRate * hiddenDeltas[j] * inputs[i]
                }
            }
            for (j in hiddenDeltas.indices) {
                biasHidden[j] += learningRate * hiddenDeltas[j]
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Инициализация сети с learning rate 0.5 и добавленными смещениями
        val nn = NeuralNetwork(2, 2, 1, 0.1)
        // Корректные данные для XOR (без дубликатов)
        val trainingData = listOf(
            Pair(doubleArrayOf(1.0, 1.0), 0.0),
            Pair(doubleArrayOf(1.0, 0.0), 1.0),
            Pair(doubleArrayOf(0.0, 1.0), 1.0),
            Pair(doubleArrayOf(0.0, 0.0), 0.0)
        )

        val trainButton = findViewById<Button>(R.id.trainButton)
        val resultTextView = findViewById<TextView>(R.id.resultTextView)

        trainButton.setOnClickListener {
            repeat(50000) {
                trainingData.shuffled().forEach { (inputs, target) ->
                    nn.train(inputs, target)
                }
            }

            val results = buildString {
                trainingData.forEach { (inputs, target) ->
                    val output = nn.forward(inputs)
                    append(
                        "Вход: ${inputs.joinToString()} -> Вывод НС: %.2f (Ожидание: %.1f)\n".format(
                            output,
                            target
                        )
                    )
                }
            }
            resultTextView.text = results
        }
    }
}