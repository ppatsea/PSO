""" 4.1 → Χρήση μεθόδου Center PSO για την εφαρμογή σε μια
          σειρά από συναρτήσεις βελτιστοποίησης. """
import numpy as np
import matplotlib.pyplot as plt

# Ορισμός Συναρτήσεν Βελτιστοποίησης
# Τετραγωνική Συνάρτηση → x^2
def function1(x):
    return np.sum(x**2)


# Τετραγωνική Συνάρτηση → (x-2)^2
def function2(x):
    return np.sum((x - 2)**2)


# Συνάρτηση Απόλυτης Τιμής → |x|
def function3(x):
    return np.sum(np.abs(x))


# Συνάρτηση Rosenbrock
def Rosenbrock(x):
    return np.sum(100 * (x[1]-x[0]**2) + (x[0]-1)**2)


# Συνάρτηση Rastrigrin
def Rastrigrin(x):
    D = len(x)
    return 10 * D + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


# Συνάρτηση Griewank
def Griewank(x):
    x = np.atleast_1d(x)
    D = x.shape[0]
    sum_term = np.sum(x**2) / 4000
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, D + 1)).reshape(-1, 1)), axis=0)
    return 1 + sum_term - prod_term


# Αρχικοποίηση του Particle Swarm
def initialize_particle_swarm(N, D):
    swarm = {
        'X': np.random.rand(N, D),  # Θέσεις Σωματιδίων
        'V': np.zeros((N, D)),      # Ταχύτητες Σωματιδίων
        'P': None,                  # Personal Καλύτερη Θέση (Αρχικοποίηση → None)
        'G': None                   # Global Καλύτερη Θέση (Αρχικοποίηση → None)
    }

    # Αρχικοποίηση προσωπικών & παγκόσμιων καλύτερων θέσεων
    swarm['P'] = np.copy(swarm['X'])
    swarm['G'] = np.copy(swarm['X'][0])
    
    return swarm


# Υλοποίηση συνάρτησης evaluate_fitness
def evaluate_fitness(X):
    if function1:
        return np.sum(X**2)
    
    elif function2:
        return np.sum((X - 2)**2)
    
    elif function3:
        return np.sum(np.abs(X))
    
    elif Rosenbrock:
        return np.sum(100 * (X[1]-X[0]**2) + (X[0]-1)**2)
    
    elif Rastrigrin:
        D = len(X)
        return 10 * D + np.sum(X**2 - 10 * np.cos(2 * np.pi * X))
    
    else:
        X = np.atleast_1d(X)
        D = X.shape[0]
        sum_term = np.sum(X**2) / 4000
        prod_term = np.prod(np.cos(X / np.sqrt(np.arange(1, D + 1)).reshape(-1, 1)), axis=0)
        return 1 + sum_term - prod_term


# Υπολογσμός νέας ταχύτητας του κάθε σωματιδίου
def update_velocity(V, X, P, G, w, c1, c2):
    r1, r2 = np.random.rand(), np.random.rand()
    return w * V + c1 * r1 * (P - X) + c2 * r2 * (G - X)


# Υλοποίηση συνάρτησης limit_velocity
def limit_velocity(V, max_velocity):
    # Περιορισμός ταχύτητας σε μία μέγιστη τιμή
    V[V > max_velocity] = max_velocity
    V[V < -max_velocity] = -max_velocity
    return V


# Υλοποίηση συνάρτησης update_position
def update_position(X, V):
    return X + V


# Υπολογισμός κέντρου κάθε συνιστώσας των θέσεων όλων των σωματιδίων
def center_particle_position(X):
    return np.mean(X, axis=0)


# Εκτέλεση PSO για τη βελτιστοποίηση της συνάρτησης
def center_pso(N, D, max_iterations, w, c1, c2, max_velocity, optimize_function):
    swarm = initialize_particle_swarm(N, D)
    positions_history = []  # Προσθήκη λίστας για αποθήκευση των θέσεων των σωματιδίων

    for t in range(max_iterations):
        fitness = optimize_function(swarm['X'])

        # Ενημέρωση personal and global καλύτερης θέσης
        update_personal_and_global_best(swarm, fitness, optimize_function)

        # Αποθήκευση των θέσεων των σωματιδίων
        positions_history.append(np.copy(swarm['X']))

        for i in range(N):
            swarm['V'][i] = update_velocity(swarm['V'][i], swarm['X'][i], swarm['P'][i], swarm['G'], w, c1, c2)
            swarm['V'][i] = limit_velocity(swarm['V'][i], max_velocity)
            swarm['X'][i] = update_position(swarm['X'][i], swarm['V'][i])

            # Ενημέρωση personal and global καλύτερης θέσης, εφόσον είναι απαραίτητο
            update_personal_and_global_best(swarm, optimize_function(swarm['X']), optimize_function)

        # Ενημέρωση της θέσης του κεντρικού σωματιδίου
        swarm['G'] = center_particle_position(swarm['X'])

        # Τερματιμός, εάν η συνολική λύση ανταποκρίνεται καλύτερα στις απαιτήσεις του προβλήματος
        if termination_condition(swarm['G']):
            break
        
    # Επιστροφή του ιστορικού των θέσεων
    return positions_history


# Υπολογισμός & ενημέρωση προσωπικών & παγκόσμιων καλύτερων θέσεων του σμήνους (swarm)
def update_personal_and_global_best(swarm, fitness, optimize_function):
    if fitness is not None:
        if np.isscalar(fitness):
            # Αν η fitness είναι μόνο ένας αριθμός (scalar)
            # Αντί να χρησιμοποιείτε len, ελέγξτε αν είναι scalar
            # ή χρησιμοποιήστε κάποια άλλη κατάλληλη συνθήκη για την επιλογή πλήθους
            pass

        else:
            for i in range(len(fitness)):
                if swarm['P'] is None or optimize_function(fitness[i]) < optimize_function(evaluate_fitness(swarm['P'][i])):
                   swarm['P'][i] = swarm['X'][i].copy()
                
                if swarm['G'] is None or optimize_function(fitness[i]) < optimize_function(evaluate_fitness(swarm['G'])):
                   swarm['G'] = swarm['X'][i].copy()


# Συνθήκη τερματισμού της PSO
def termination_condition(global_best, threshold_fitness=1e-5, max_iterations=1000):
    if evaluate_fitness(global_best) < threshold_fitness or max_iterations <= 0:
        return True
    else:
        return False

# Παράμετροι & συναρτήσεις που χρησιμοποιούνται για την εκτέλεση της PSO 
N = 10                  # Αριθμός Σωματιδίων (particles)
D = 2                   # Διάσταση του προβλήματος, ή πλήθος μεταβλητών που πρέπει να βελτιστοποιηθούν
max_iterations = 100    # Αριθμός Επαναλήψεων ή εποχών
w = 0.5                 # Βάρος Αδράνειας (Inertia weight)
c1 = 1.5                # Συντελεστής ενημέρωσης για την προσωπική καλύτερη θέση
c2 = 1.5                # Συντελεστής ενημέρωσης για τη γενική καλύτερη θέση 
max_velocity = 0.2      # Μέγιστος επιτρεπτός ρυθμός αλλαγής των θέσεων των σωματιδίων

# Καθορισμός διαφορετικών χρωμάτων για κάθε συνάρτηση
colors = ['red', 'green', 'blue', 'purple', 'black', 'orange']

# functions_to_optimize = [function1, function2, function3, Rosenbrock, Rastrigrin, Griewank]
functions_to_optimize = [Rosenbrock, Rastrigrin, Griewank]


# Εμφάνιση όλων των θέσεων κάθε σωματιδίου
# for i, func in enumerate(functions_to_optimize):
#     positions_history = center_pso(N, D, max_iterations, w, c1, c2, max_velocity, func)
#     best_solution = positions_history[-1]  # Η τελευταία θέση είναι η βέλτιστη λύση

#     print(f"Βέλτιστη Λύση για την {func.__name__}: {best_solution}\n")
#     print(positions_history)

#     # Εμφάνιση των θέσεων των σωματιδίων στον χώρο των μεταβλητών
#     positions_history = np.array(positions_history)
#     plt.plot(positions_history[:, :, 0], positions_history[:, :, 1], 'o-', label=f"{func.__name__}", color=colors[i])
#     plt.scatter(best_solution[0], best_solution[1], color=colors[i])  # Εμφάνιση της βέλτιστης λύσης με το αντίστοιχο χρώμα


# Εμφάνιση μόνο της τελευταίας θέσης κάθε σωματιδίου
for i, func in enumerate(functions_to_optimize):
    positions_history = center_pso(N, D, max_iterations, w, c1, c2, max_velocity, func)
    if positions_history and len(positions_history) > 0 and len(positions_history[-1]) > 0:
        # Εάν υπάρχει τουλάχιστον μια εγγραφή στο ιστορικό για την τρέχουσα συνάρτηση
        best_solution = positions_history[-1][-1]  # Η τελευταία θέση του τελευταίου σωματιδίου είναι η βέλτιστη λύση

        print(f"Βέλτιστη Λύση για την {func.__name__}: {best_solution}\n")
    
        # Έλεγχος για το μέγεθος της λίστας colors πριν από την πρόσβαση
        if i < len(colors):
            # Εμφάνιση της τελευταίας θέσης κάθε σωματιδίου σε διαφορετικό χρώμα
            plt.plot(positions_history[-1][:, 0], positions_history[-1][:, 1], 'o-', label=f"{func.__name__}", color=colors[i])
            plt.scatter(best_solution[0], best_solution[1], color=colors[i])  # Εμφάνιση της βέλτιστης λύσης με το αντίστοιχο χρώμα


# Προσθήκη ετικετών στο plot
plt.title("Optimization Functions")     # Λειτουργίες Βελτιστοποίησης
plt.xlabel("x")
plt.ylabel("Function Value")            # Αξία συνάρτησης
plt.legend()
plt.show()