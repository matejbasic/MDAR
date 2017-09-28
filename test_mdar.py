import time
from mdar.recommender import MDAR
from mdar.query_manager import QueryManager
from mdar.data_manager import DataManager
from mdar.tester import Tester
from mdar.results import Results

k_fold_size = 4
config_path = 'config.json'

available_approaches = [
    ('order_association', 1),
    ('user_history', 1),
    ('user_history2', 1),
    ('time_related', 1),
]

tester = Tester()

dmrec = []
for i in range(0, k_fold_size):
    dm  = DataManager(config_path, k_fold_size)
    dm.set_testing_part_index(i)

    rec = MDAR(used_approaches = available_approaches)
    rec.set_data_manager(dm)

    start = time.time()
    rec.train(100)
    print 'training time:\t' + str(time.time() - start)
    dmrec.append({'dm': dm, 'rec': rec})

for k in [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 75, 100]:
    print 'recommending with k=' + str(k) + ' items'
    tester.set_k(k)
    results = Results()
    for i in range(0, k_fold_size):
        tester.set_data_manager(dmrec[i]['dm'])
        tester.set_recommender(dmrec[i]['rec'])

        start = time.time()
        precision, recall, fallout, f1, specificity, confusion_matrix, cases_without_history = tester.test()
        print 'recommendation testing time for k=' + str(k) + ': ' + str(time.time() - start)

        results.add(precision, recall, fallout, f1, specificity, confusion_matrix, cases_without_history)

    print results.get_confusion_matrix()
    p, r, f, f1, s = results.get_metrics()
    print 'model\t k\t precision\t\t recall\t\t fallout\t\t f1\t\t specificity'
    print 'MDAR', '\t', k, '\t', p, '\t', r, '\t', f, '\t', f1, '\t', s
    print '-' * 22

    results.log_results('MDAR', k, dm.get_items_count('all'))
