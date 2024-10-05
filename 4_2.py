""" 4.2 → Χρήση μεθόδου PSO για την εκπαίδευση τεχνητών νευρωνικών 
          δικτύων για προβλήματαnμάθησης συναρτήσεων. """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

# Ορισμός Συναρτήσεν Βελτιστοποίησης
def f(x):
    # # Τετραγωνική Συνάρτηση → x^2
    # return np.sum(x**2)


    # # Τετραγωνική Συνάρτηση → (x-2)^2
    # return np.sum((x - 2)**2)


    # # Συνάρτηση Απόλυτης Τιμής → |x|
    # return np.sum(np.abs(x))


    # # Rosenbrock Function
    # return np.sum(100 * (x[1]-x[0]**2) + (x[0]-1)**2)


    # # Rastrigrin Function
    # D = len(x)
    # return 10 * D + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
    

    # Griewank Function
    x = np.atleast_1d(x)
    D = x.shape[0]
    sum_term = np.sum(x**2) / 4000
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, D + 1)).reshape(-1, 1)), axis=0)
    return 1 + sum_term - prod_term


# Δημιουργία τυχαίων δεδομένων εκπαίδευσης
np.random.seed(42)  # Για αναπαραγωγή των ίδιων αποτελεσμάτων
x_train = np.random.uniform(-1, 1, (100, 2))
y_train = np.apply_along_axis(f, 1, x_train.reshape(-1, 2, order='F')).reshape(-1, 1)


# Δημιουργία μοντέλου Νευρωνικού Δικτύου
def create_model(input_shape):
    model = Sequential()
    model.add(Dense(1, activation='linear', input_shape=input_shape))
    model.compile(loss='mse', optimizer='adam')
    return model


# Υλοποίηση Αλγορίθμου PSO για την εκπαίδευση των ΑΝΝ
def PSO(x_train, y_train, epochs, pop_size, w, c1, c2):
    particles = np.random.uniform(-1, 1, (pop_size, 2))
    models = [create_model((2,)) for _ in range(pop_size)]

    for epoch in range(epochs):
        for i, particle in enumerate(particles):
            particle[0] = particle[0] * w + c1 * np.random.rand() * (particles[np.argmin(particles[:, 1]), 0] - particle[0]) + c2 * np.random.rand() * (particles[np.argmax(particles[:, 1]), 0] - particle[0])
            particle[1] = particle[1] * w + c1 * np.random.rand() * (particles[np.argmin(particles[:, 1]), 1] - particle[1]) + c2 * np.random.rand() * (particles[np.argmax(particles[:, 1]), 1] - particle[1])

            model = models[i]
            model.fit(x_train, y_train[:100], epochs=10, batch_size=32)
            particles[i, 1] = model.evaluate(x_train, y_train[:100], verbose=0)

    best_particle_index = np.argmin(particles[:, 1])
    best_particle = particles[best_particle_index, :]
    best_model = models[best_particle_index]

    return best_particle, best_model


if __name__ == '__main__':
    x_train = np.random.uniform(-1, 1, (100, 2))
    y_train = np.apply_along_axis(f, 1, x_train)
    y_train = y_train.reshape(-1, 1)
    y_train = y_train[:100]

    x = np.arange(-1, 1, 0.01)
    y = f(x)
    x_test = np.column_stack((x, x**2))
    y_test = np.apply_along_axis(f, 1, x_test)
    y_test = y_test.reshape(-1, 1)

    # Κλήση της PSO
    best_particle, best_model = PSO(x_train, y_train, epochs=10, pop_size=10, w=0.5, c1=1.5, c2=1.5)

    # Εκτύπωση των σχημάτων για τα δεδομένα εκπαίδευσης και ελέγχου
    # plt.title('Τετραγωνική Συνάρτηση → x^2')
    # plt.title('Τετραγωνική Συνάρτηση → (x-2)^2')
    # plt.title('Συνάρτηση Απολύτου x → |x|')
    # plt.title('Συνάρτηση Rosenbrock')
    # plt.title('Συνάρτηση Rastrigrin')
    plt.title('Συνάρτηση Griewank')

    plt.plot(x_train[:, 0], y_train, label='Training Data')
    plt.plot(x_test[:, 0], y_test[:200], label='Test Data')
    plt.plot(x_train[:, 0], best_model.predict(x_train), label='PSO Predictions')
    plt.legend()
    plt.show()