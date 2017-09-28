# MDAR

Multidimensional Association Recommender based on association analysis and graph database (Neo4j).

## Structure
Recommender is main class used for training (based on ARHR and mode) and generating recommendations.

Results class aggregates testing results.

Tester class is used for testing recommendations and calculating IR metrics such as precision, recall, F1 and other.

QueryManager class is used for communicating with Neo4j graph database and constructing TF nodes constraints for test and train dataset parts (k-fold cross validation).

DataManager class inherits QueryManager and it's used for fetching data and transforming it in to appropriate format for further usage.

### Recommenders
BaseRecommender class acts as a base for other recommender classes with min support, confidence and lift.

AssociationRecommender support confidence, lift and support metrics with time constraints, >1 degree recommendations and multiple items in body and head of association rule.

OrderAssociationRecommender, inherits AssociationRecommender, uses current cart items as body items (recommendations source).

UserHistoryRecommender, inherits AssociationRecommender, uses previously purchased items as a source.

TimeRelatedRecommender returns recommendations based on given time constraints. Fallbacks on global popular items if none.


## Other

Part of master thesis "Recommender systems based on association analysis" (mentor: prof.dr.sc Božidar Kliček) @ FOI, Varaždin.

Other thesis related projects:
- [recommenders playground](https://github.com/matejbasic/recommenders-playground)
- [used datasets](https://github.com/matejbasic/recomm-ecommerce-datasets)
