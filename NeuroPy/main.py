import numpy
import scipy.special

# Определение класса нейронной сети
class neuralNetwork:

    # инициализировать нейронную сеть
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # Матрицы весовых коэффициентов связей wih и who.
        # Весовые коэффициенты связей между узлом i и узлом j
        # следующего слоя обозначены как w__i__j:
        # wll w21
        # wl2 w22 и т.д.
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # коэффициент обучения
        self.lr = learningrate

        # использование сигмоиды в качестве функции активации
        self.activation_function = lambda x: scipy.special.expit(x)

    def train():
        pass

    # опрос нейронной сети
    def query(self, inputs_list):
        # преобразовать список входных значений
        # в двухмерный массив
        print('Входные данные: \n',inputs_list)
        inputs = numpy.array(inputs_list, ndmin=2).T
        print('Преобразуем в столбец: \n',inputs)

        # рассчитать входящие сигналы для скрытого слоя
        hidden_inputs = numpy.dot(self.wih, inputs)
        print('Входные данные для скрытого слоя: \n', hidden_inputs)

        # рассчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)
        print('Применим функцию активации к входным данным скрытого слоя: \n', hidden_outputs)

        # рассчитать входящие сигналы для выходного слоя
        final_inputs = numpy.dot(self.who, hidden_outputs)
        print('Входные данные для выходного слоя: \n', final_inputs)

        # рассчитать исходящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)
        print('РЕЗУЛЬТАТ: \n', final_outputs)

        return final_outputs

pass

#mat1 = numpy.random.normal(0.0,0.3,(2,5))

neuro = neuralNetwork(3,3,3,0.25)
res = (neuro.query([1,0.5,-1.5]))



