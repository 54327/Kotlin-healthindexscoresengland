import smile.data.DataFrame
import smile.data.formula.formula
import smile.io.Read.csv
import smile.regression.ols
import kotlin.math.round

fun main() {
    // 读取 CSV 数据
    val filePath = "D:\Desktop\health_data\healthindexscoresengland.csv"
    val df = csv(filePath)

    // 显示前几行数据
    println(df.summary())

    // 选取特征和目标变量
    val featureCols = arrayOf("GDP_Spending", "LifeExpectancy", "PollutionIndex", "SmokingRate")
    val targetCol = "HealthIndex"

    // 确保数据格式正确
    val data = df.select(*featureCols, targetCol).toArray()
    val X = data.map { it.sliceArray(0 until featureCols.size) }.toTypedArray()
    val y = data.map { it.last() }.toDoubleArray()

    // 训练线性回归模型
    val model = ols(X, y)

    // 输出模型的回归系数
    println("模型回归系数: ${model.coefficients().contentToString()}")

    // 未来预测数据（假设未来年份的 GDP、寿命、污染指数等）
    val futureYears = arrayOf(2025, 2026, 2027, 2028, 2029, 2030)
    val futureX = arrayOf(
        doubleArrayOf(10.5, 82.5, 28.0, 7.5),
        doubleArrayOf(10.8, 82.7, 27.0, 7.2),
        doubleArrayOf(11.0, 82.9, 26.0, 7.0),
        doubleArrayOf(11.3, 83.1, 25.0, 6.8),
        doubleArrayOf(11.5, 83.3, 24.0, 6.5),
        doubleArrayOf(11.8, 83.5, 23.0, 6.2)
    )

    // 预测未来健康指数
    val futurePredictions = futureX.map { round(model.predict(it) * 100) / 100 }  // 保留两位小数

    // 输出预测结果
    futureYears.zip(futurePredictions).forEach { (year, prediction) ->
        println("预计 $year 年健康指数: $prediction")
    }
}