from typing import Dict
from collections import Counter
from flowers import Bouquet, Flower, FlowerSizes, FlowerColors, FlowerTypes
from suitors.base import BaseSuitor
import random
import numpy as np
from constants import MAX_BOUQUET_SIZE
from utils import flatten_counter
import itertools
import math


class Suitor(BaseSuitor):
    def __init__(self, days: int, num_suitors: int, suitor_id: int):
        """
        :param days: number of days of courtship
        :param num_suitors: number of suitors, including yourself
        :param suitor_id: unique id of your suitor in range(num_suitors)
        """
        super().__init__(days, num_suitors, suitor_id, name='g8')
        if self.days > 1:
            self.type_weights = [1/(self.days**2), 1/(self.days), 2/(self.days), 1]
            self.color_weights = [1/(self.days**2), 1/(self.days**2), 1/(self.days), 1/(self.days), 2/(self.days), 1]
            self.size_weights = [1/(self.days**2), 1/(self.days), 1]
            random.shuffle(self.type_weights)
            random.shuffle(self.color_weights)
            random.shuffle(self.size_weights)
            # pick one attribute to null
            self.null = random.randint(0,2) # 0=type, 1=color, 2=size
            self.adjust_weights_number()
        elif self.days==1:
            self.alloptions = []
            for n in range(1, 13):
                self.alloptions = self.alloptions + list(itertools.combinations_with_replacement([0,1,2,3,4,5], n))
            l = len(self.alloptions)
            ones = math.ceil((1/(self.num_suitors-1))*l)
            zeros = l - ones
            mask = [True for n in range(ones)] + [False for n in range(zeros)]
            random.shuffle(mask)
            self.alloptions = list(itertools.compress(self.alloptions, mask))
            print(self.alloptions)
            self.null = 3 # so its ignored by scoring methods
        self.given = {} # dictionary to save the bouquets we gave + their scores
        all_ids = np.arange(self.num_suitors)
        self.recipient_ids = all_ids[all_ids != self.suitor_id]
        for suitor in self.recipient_ids:
            self.given[suitor] = [] # list for each suitor, will be list of lists with [bouquet, score]
        self.days_left = days
        self.day_number = 0
        self.initalize_controlled_strat()
        self.suitors_to_test = list(self.recipient_ids)
        # Dictionary to remember if we receive score > 0.95 and rank 1 for any bouquet given
        # Will be a dictionary with key as recipient_id and value as a list of lists with bouquets and scores
        self.remember_high_scores = {}
        # Remember number of high scores we stored in self.remember_high_scores
        # Will be a dictionary with key as recipient_id and value as number of scores we stored
        self.num_high_scores = {}
        for suitor in self.recipient_ids:
            self.remember_high_scores[suitor] = []
            self.num_high_scores[suitor] = 0
        # Remember bouquets given this round
        self.bouquets_given_this_round = {}

        # Priority to make bouquets in last round. Dictionary of 2 lists.
        # First list accessed with key "saved_set", second with "rank".
        self.priority = {}

    def get_all_possible_bouquets_size_6(self, flowers: Dict[Flower, int]):
        flat_flower = flatten_counter(flowers)
        bouquets = []
        for size in range(1, 7):
            size_bouquets = list(set(list(itertools.combinations(flat_flower, size))))
            bouquets += size_bouquets
        return bouquets


    def _prepare_bouquet(self, remaining_flowers, recipient_id):
        num_remaining = sum(remaining_flowers.values())
        size = int(np.random.randint(0, min(MAX_BOUQUET_SIZE, num_remaining) + 1))
        if size > 0:
            chosen_flowers = np.random.choice(flatten_counter(remaining_flowers), size=(size, ), replace=False)
            chosen_flower_counts = dict(Counter(chosen_flowers))
            for k, v in chosen_flower_counts.items():
                remaining_flowers[k] -= v
                assert remaining_flowers[k] >= 0
        else:
            chosen_flower_counts = dict()
        chosen_bouquet = Bouquet(chosen_flower_counts)
        return self.suitor_id, recipient_id, chosen_bouquet

    def best_bouquets(self, remaining_flowers, scores_per_player):
        chosen_bouquets = {}
        suitor_bouquet_counts = {}

        for suitor in self.recipient_ids:
            chosen_bouquets[suitor] = {}
            suitor_bouquet_counts[suitor] = 0

        # Look at saved high-scoring bouquets and see if you can make the same bouquet
        for suitor_data in self.priority["saved_set"]:
            i = 0
            while(i < suitor_data[1]):
                # Get saved bouquet and get its score per attr
                bouquet = self.remember_high_scores[suitor_data[0]][i][0]
                #print("in best_bouquet")
                #print(bouquet)
                score_attr = self.get_score_per_attr(bouquet, "bouquet")
                #print("in best_bouquet, returned from get_score_per_attr")
                all_possible_bouquets = self.get_all_possible_bouquets_size_6(remaining_flowers)
                # all_possible_bouquets is a list of tuples
                #print(all_possible_bouquets)
                found = 0
                for possible_bouquet in all_possible_bouquets:
                    #print(possible_bouquet)
                    score_attr_possible_bouquet = self.get_score_per_attr(possible_bouquet, "tuple")
                    if(score_attr_possible_bouquet == None):
                        continue
                    if(score_attr == score_attr_possible_bouquet): # found an exact match
                        found = 1
                        break
                # update remaining_flowers
                if found:
                    d1 = Counter(bouquet.flowers())
                    d2 = Counter(possible_bouquet)
                    d3 = d1 - d2
                    remaining_flowers = dict(d3)
                    chosen_bouquets[suitor_data[0]] = dict(d2)
                    suitor_bouquet_counts[suitor_data[0]] = len(possible_bouquet)
                    break

                i = i + 1


        flowers_in_market = flatten_counter(remaining_flowers)

        for flower in flowers_in_market:
            color = flower.color
            type = flower.type
            size = flower.size
            max_score=0
            suitor_getting_flower=0
            max_rank = self.num_suitors
            for suitor in self.recipient_ids:
                color_score = scores_per_player[suitor]["color"][color.value]
                type_score = scores_per_player[suitor]["type"][type.value]
                size_score = scores_per_player[suitor]["size"][size.value]
                total_score = color_score+type_score+size_score

                if total_score>max_score and suitor_bouquet_counts[suitor]<7 and self.priority["rank"][suitor] < max_rank:
                    max_score=total_score
                    suitor_getting_flower=suitor
            if flower in chosen_bouquets[suitor_getting_flower]:
                chosen_bouquets[suitor_getting_flower][flower] = chosen_bouquets[suitor_getting_flower][flower]+1
            else:
                chosen_bouquets[suitor_getting_flower][flower] = 1
            suitor_bouquet_counts[suitor_getting_flower] = suitor_bouquet_counts[suitor_getting_flower] + 1

        return chosen_bouquets

    def convert_prefs_to_scores_per_player(self):
        scores_by_attribute = {}
        colors = [c.name for c in FlowerColors]
        sizes = [s.name for s in FlowerSizes]
        types = [t.name for t in FlowerTypes]
        for suitor, prefs in self.prefs.items():
            preferences = {}
            preferences["color"] = [0]*6
            preferences["type"] = [0]*4
            preferences["size"] = [0]*3
            for param, val in prefs.items():
                if param in colors:
                    i = colors.index(param)
                    preferences["color"][i] = val
                    continue
                if param in types:
                    i = types.index(param)
                    preferences["type"][i] = val
                    continue
                if param in sizes:
                    i = sizes.index(param)
                    preferences["size"][i] = val
            # TODO: Can modify this. Normalizing the scores after controlled experiment
            preferences["color"] = [float(i)/(max(preferences["color"]) + 1e-6) for i in preferences["color"]]
            preferences["size"] = [float(i) / (max(preferences["size"]) + 1e-6) for i in preferences["size"]]
            preferences["type"] = [float(i) / (max(preferences["type"]) + 1e-6) for i in preferences["type"]]
            scores_by_attribute[suitor] = preferences
        print(scores_by_attribute)
        return scores_by_attribute

    def scores_per_player(self, given):
        scores_by_attribute = {} # will be a dictionary of dictionaries, one per suitor, dictionaries contain info on scores per attribute
        for suitor in given:
            preferences = {}
            preferences["color"] = [[],[],[],[],[],[]]
            preferences["type"] = [[],[],[],[]]
            preferences["size"] = [[],[],[]]
            preferences["number"] = [[],[],[],[],[],[],[],[],[],[],[],[],[]]
            list_of_bouquets_and_scores = given[suitor]
            # first, list of scores for every time a specific attribute occured
            for round in list_of_bouquets_and_scores:
                bouquet = round[0]
                score = round[1]
                number = len(bouquet)
                preferences["number"][number].append(score)
                for flower in bouquet.flowers():
                    preferences["color"][flower.color.value].append(score)
                    preferences["type"][flower.type.value].append(score)
                    preferences["size"][flower.size.value].append(score)

            # next, average the lists so we know average score per attribute
            for i,color in enumerate(preferences["color"]):
                if len(preferences["color"][i]) != 0:
                    preferences["color"][i] = sum(preferences["color"][i])/len(preferences["color"][i])
                else:
                    preferences["color"][i] = 0
            for i,type in enumerate(preferences["type"]):
                if len(preferences["type"][i]) != 0:
                    preferences["type"][i] = sum(preferences["type"][i]) / len(preferences["type"][i])
                else:
                    preferences["type"][i] = 0
            for i,size in enumerate(preferences["size"]):
                if len(preferences["size"][i]) != 0:
                    preferences["size"][i] = sum(preferences["size"][i]) / len(preferences["size"][i])
                else:
                    preferences["size"][i] = 0
            for i,n in enumerate(preferences["number"]):
                if len(preferences["number"][i]) != 0:
                    preferences["number"][i] = sum(preferences["number"][i]) / len(preferences["number"][i])
                else:
                    preferences["number"][i] = 0

            scores_by_attribute[suitor]=preferences

        return scores_by_attribute


    def get_average_rank(self, given):
        priority = {}
        for suitor in given:
            sum_rank = 0
            list_of_bouquets_and_scores = given[suitor]
            for round in list_of_bouquets_and_scores:
                sum_rank += round[2]
            avg_rank = sum_rank / len(list_of_bouquets_and_scores)

            priority[suitor] = avg_rank

        return priority


    def get_score_per_attr(self, bouquet, typeofdata):
        attr_score = {
            "Small" : 0,
            "Medium" : 0,
            "Large" : 0,
            "White" : 0,
            "Yellow" : 0,
            "Red" : 0,
            "Purple" : 0,
            "Orange" : 0,
            "Blue" : 0,
            "Rose" : 0,
            "Chrysanthemum" : 0,
            "Tulip" : 0,
            "Begonia" : 0,
            "None" : 0
        }
	
        #print(type(bouquet))
        if(typeofdata == "tuple"):
            iteratorobj = bouquet
        else:
            iteratorobj = bouquet.flowers()
        
        #if(type(bouquet) == tuple):
        #    print("---------------------------------------tuple found, return ----------------------------------------")
        #    return None
        
        for flower in iteratorobj:
            if flower.size == FlowerSizes.Small:
                attr_score["Small"] += 1
            elif flower.size == FlowerSizes.Medium:
                attr_score["Medium"] += 1
            else:
                attr_score["Large"] += 1

            if flower.type == FlowerTypes.Rose:
                attr_score["Rose"] += 1
            elif flower.type == FlowerTypes.Tulip:
                attr_score["Tulip"] += 1
            elif flower.type == FlowerTypes.Begonia:
                attr_score["Begonia"] += 1
            else:
                attr_score["Chrysanthemum"] += 1

            if flower.color == FlowerColors.Red:
                attr_score["Red"] += 1
            elif flower.color == FlowerColors.Blue:
                attr_score["Blue"] += 1
            elif flower.color == FlowerColors.White:
                attr_score["White"] += 1
            elif flower.color == FlowerColors.Purple:
                attr_score["Purple"] += 1
            elif flower.color == FlowerColors.Orange:
                attr_score["Orange"] += 1
            else:
                attr_score["Yellow"] += 1

        return attr_score


    def prepare_bouquets(self, flower_counts: Dict[Flower, int]):

        """:param flower_counts: flowers and associated counts for for available flowers
        :return: list of tuples of (self.suitor_id, recipient_id, chosen_bouquet)
        the list should be of length len(self.num_suitors) - 1 because you should give a bouquet to everyone
         but yourself

        To get the list of suitor ids not including yourself, use the following snippet:"""

        if self.days > 30:
            self.day_number += 1
            if self.day_number == self.days:
                scores_per_player = self.convert_prefs_to_scores_per_player()
                # scores_per_player is already excluding us, and is in order of the other suitors (index = suitor)
                chosen_bouquets = self.best_bouquets(flower_counts.copy(), scores_per_player)
                bouquets = []
                for suitor in chosen_bouquets:
                    bouquets.append((self.suitor_id, suitor, Bouquet(chosen_bouquets[suitor])))
                # print(str(self.suitor_id) + str(self.size_weights) +str(self.type_weights) + str(self.color_weights))
                return bouquets
            return self.use_controlled_strategy(flower_counts.copy())


        # this loop inputs the score for each bouquet we gave last round
        if len(self.feedback)!=0:
            this_rounds_feedback = self.feedback[-1]
            for i,rank_score in enumerate(this_rounds_feedback): # rank_score is tuple with rank,score,ties
                if i!=self.suitor_id:
                    score = rank_score[1]
                    suitor = i
                    self.given[suitor][-1][1] = score
                    self.given[suitor][-1][2] = rank_score[0]

                    if score >= 0.95 and rank_score[0] == 1:   # Remember this bouquet
                        temp = []
                        temp.append(self.bouquets_given_this_round[suitor])
                        temp.append(score)
                        self.remember_high_scores[suitor].append(temp)
                        self.num_high_scores[suitor] = self.num_high_scores[suitor] + 1

            print("-------------------------------------------------------------------------------------------------------------")
            print(self.remember_high_scores)



        all_ids = np.arange(self.num_suitors)
        recipient_ids = all_ids[all_ids != self.suitor_id]
        remaining_flowers = flower_counts.copy()


        # on the last day prepare special bouquets
        if self.days_left==1:
            # list of (self.suitor_id, recipient_id, chosen_bouquet)
            scores_per_player = self.scores_per_player(self.given) # list of dictionaries {number: [number preferences], color: [color preferences], etc.}
            # scores_per_player is already excluding us, and is in order of the other suitors (index = suitor)
            # Sort who has highest number of scores saved for priority class "saved_set"
            self.priority["saved_set"] = sorted(self.num_high_scores.items(), key = lambda x : x[1])
            # Sort who has highest number of scores saved for priority class "rank"
            #self.priority["rank"] = sorted(self.get_average_rank(self.given).items(), key = lambda x : x[1])
            self.priority["rank"] = self.get_average_rank(self.given)

            chosen_bouquets = self.best_bouquets(remaining_flowers, scores_per_player)
            bouquets = []
            for suitor in chosen_bouquets:
                bouquets.append((self.suitor_id, suitor, Bouquet(chosen_bouquets[suitor])))
        else:
            # this loop saves the bouquets so next round we can see the scores
            bouquets = list(
                map(lambda recipient_id: self._prepare_bouquet(remaining_flowers, recipient_id), recipient_ids))
            for bouquet in bouquets:
                player_given_to = bouquet[1]
                actual_bouquet = bouquet[2]
                self.given[player_given_to].append([actual_bouquet, 0, 0])

                self.bouquets_given_this_round[player_given_to] = actual_bouquet
            self.days_left=self.days_left-1



        return bouquets

    def zero_score_bouquet(self):
        """
        :return: a Bouquet for which your scoring function will return 0
        """
        return Bouquet({})

    def one_score_bouquet(self):
        """
        :return: a Bouquet for which your scoring function will return 1
        """
        max_flower = Flower(
            size=FlowerSizes(self.size_weights.index(1)),
            color=FlowerColors(self.color_weights.index(1)),
            type=FlowerTypes(self.type_weights.index(1))
        )
        return Bouquet({max_flower: 12})

    def score_types(self, types: Dict[FlowerTypes, int]):
        """
        :param types: dictionary of flower types and their associated counts in the bouquet
        :return: A score representing preference of the flower types in the bouquet
        """
        if self.null == 0 or self.days == 1:
            return 0

        score = 0
        total = 0
        # sum up the scores of each flower type
        for flower in types:
            index = flower.value
            number = types[flower]
            score = score + (self.type_weights[index]*number)
            total = total + number

        # get average score for number of flowers
        if total != 0:
            score = score/total

        # multiply by .4 since each type, sixe, color is .4 weight but one is dropped + .2 number
        return score*.4

    def score_colors(self, colors: Dict[FlowerColors, int]):
        """
        :param colors: dictionary of flower colors and their associated counts in the bouquet
        :return: A score representing preference of the flower colors in the bouquet
        """
        if self.null == 1:
            return 0

        # color only one if days = 1 --> Probabilisitic
        if self.days == 1:
            color_list = []
            for flower in colors:
                index = flower.value
                number = colors[flower]
                for n in range(0, number):
                    color_list.append(index)
            color_list.sort()
            color_tuple = tuple(color_list)
            if color_tuple in self.alloptions:
                return 1
            else:
                return 0

        score = 0
        total = 0
        # sum up the scores of each flower type
        for flower in colors:
            index = flower.value
            number = colors[flower]
            score = score + (self.color_weights[index] * number)
            total = total + number

        # get average score for number of flowers
        if total != 0:
            score = score / total

        # multiply by .4 since each type, sixe, color is .4 weight but one is dropped + .2 number
        return score * .4

    def score_sizes(self, sizes: Dict[FlowerSizes, int]):
        """
        :param sizes: dictionary of flower sizes and their associated counts in the bouquet
        :return: A score representing preference of the flower sizes in the bouquet
        """
        # weights
        # weights = [0, .5, 1]

        # get preference order
        # random.shuffle(weights)

        if self.days == 1:
            return 0

        score = 0
        total = 0
        # sum up the scores of each flower type
        for flower in sizes:
            index = flower.value # get enum value for flower attribute
            number = sizes[flower] # get number of flowers
            score = score + (self.size_weights[index] * number)
            total = total + number

        # get average score for number of flowers
        if total != 0:
            score = score / total

        # multiply by .4 since each type, sixe, color is .4 weight but one is dropped + .2 number
        score = score * .4

        if self.null == 2:
            score = 0

        # count number of flowers for the last .25
        # weights_number = [0, .05, .18, .32, .48, .72, .85, 1, 1, 1, 1, 1, 1]
        score_count = self.weights_number[total]*.2

        return score+score_count

    def adjust_weights_number(self):
        num_one_flowers = [4, 6, 3][self.null]
        max_flowers = 6 * (self.num_suitors - 1)
        expected_count = max(1, max_flowers // 72)
        max_bouquet_size = min(9, num_one_flowers * expected_count)
        weights_number = [0] + [1] * 12
        '''
        Use exponential curve to map weights for number of flowers in bouquet
        Want a curve such that value at max_bouquet_size is 1, and value at 0 is 0
        Let f(x) = e^(ax) - 1
        Let t = max_bouquet_size
        f(t) = 1 = e^(at) - 1 => a = ln2/t
        '''
        t = max_bouquet_size
        a = math.log(2)/t
        for i in range(1, t):
            temp = math.exp(a * i) - 1
            weights_number[i] = round(temp, 4)
        self.weights_number = weights_number
        self.personal_max_bouquet_size = max_bouquet_size

    def receive_feedback(self, feedback):
        """
        :param feedback:
        :return: nothing
        """
        self.feedback.append(feedback)

    def initalize_controlled_strat(self):
        self.prefs = {}
        self.controlled_strat = {}
        self.controlled_given = {}
        for suitor in self.recipient_ids:
            self.controlled_strat[suitor] = {
                "colors" : [c for c in FlowerColors],
                "types" : [t for t in FlowerTypes],
                "sizes" : [s for s in FlowerSizes],
                "const_params_color" : [],
                "const_params_type": [],
                "const_params_size": [],
                "tested_param" : ""
            }
            self.prefs[suitor] = {
                "Small": 0,
                "Medium": 0,
                "Large": 0,
                "White": 0,
                "Yellow": 0,
                "Red": 0,
                "Purple": 0,
                "Orange": 0,
                "Blue": 0,
                "Rose": 0,
                "Chrysanthemum": 0,
                "Tulip": 0,
                "Begonia": 0,
                "None": 0
            }
            self.controlled_given[suitor] = []

    def start_controlled_color(self, suitor, flower):
        self.controlled_strat[suitor]["colors"].remove(flower.color)
        self.controlled_strat[suitor]["const_params_color"] = [flower.size, flower.type]
        self.controlled_strat[suitor]["tested_param"] = flower.color.name

    def start_controlled_type(self, suitor, flower):
        self.controlled_strat[suitor]["types"].remove(flower.type)
        self.controlled_strat[suitor]["const_params_type"] = [flower.color, flower.size]
        self.controlled_strat[suitor]["tested_param"] = flower.type.name

    def start_controlled_size(self, suitor, flower):
        self.controlled_strat[suitor]["sizes"].remove(flower.size)
        self.controlled_strat[suitor]["const_params_size"] = [flower.color, flower.type]
        self.controlled_strat[suitor]["tested_param"] = flower.size.name

    def update_preferences(self):
        if self.day_number > 1:
            latest_feedback = self.feedback[-1]
            for suitor in self.tested_suitors:
                self.prefs[suitor][self.controlled_strat[suitor]["tested_param"]] = latest_feedback[suitor][1]

    def get_flowers_to_test(self, flowers_in_market, suitor):
        all_tested_flag = True
        for color in self.controlled_strat[suitor]["colors"]:
            all_tested_flag = False
            reqd_flower = Flower(
                size=self.controlled_strat[suitor]["const_params_color"][0],
                color=color,
                type=self.controlled_strat[suitor]["const_params_color"][1]
            )
            if reqd_flower in flowers_in_market:
                self.controlled_strat[suitor]["colors"].remove(color)
                self.controlled_strat[suitor]["tested_param"] = color.name
                return reqd_flower, all_tested_flag

        for type in self.controlled_strat[suitor]["types"]:
            all_tested_flag = False
            reqd_flower = Flower(
                size=self.controlled_strat[suitor]["const_params_type"][1],
                color=self.controlled_strat[suitor]["const_params_type"][0],
                type=type
            )
            if reqd_flower in flowers_in_market:
                self.controlled_strat[suitor]["types"].remove(type)
                self.controlled_strat[suitor]["tested_param"] = type.name
                return reqd_flower, all_tested_flag

        for size in self.controlled_strat[suitor]["sizes"]:
            all_tested_flag = False
            reqd_flower = Flower(
                size=size,
                color=self.controlled_strat[suitor]["const_params_size"][0],
                type=self.controlled_strat[suitor]["const_params_size"][1]
            )
            if reqd_flower in flowers_in_market:
                self.controlled_strat[suitor]["sizes"].remove(size)
                self.controlled_strat[suitor]["tested_param"] = size.name
                return reqd_flower, all_tested_flag
        self.controlled_strat[suitor]["tested_param"] = "None"
        return None, all_tested_flag

    def use_controlled_strategy(self, remaining_flowers):
        chosen_bouquets = {}
        flowers_in_market = flatten_counter(remaining_flowers)
        remaining_receipients = list(self.recipient_ids)
        self.update_preferences()
        self.tested_suitors = []
        for suitor in list(self.suitors_to_test):
            all_tested = False
            if self.day_number == 1:
                chosen_flower = np.random.choice(list(set(flowers_in_market)))
                self.start_controlled_color(suitor, chosen_flower)
            elif self.day_number == 2:
                chosen_flower = np.random.choice(list(set(flowers_in_market)))
                self.start_controlled_type(suitor, chosen_flower)
            elif self.day_number == 3:
                chosen_flower = np.random.choice(list(set(flowers_in_market)))
                self.start_controlled_size(suitor, chosen_flower)
            else:
                chosen_flower, all_tested = self.get_flowers_to_test(flowers_in_market, suitor)

            if chosen_flower is not None:
                chosen_bouquets[suitor] = chosen_flower
                flowers_in_market.remove(chosen_flower)
                remaining_receipients.remove(suitor)
                self.tested_suitors.append(suitor)
            if all_tested:
                self.suitors_to_test.remove(suitor)

        bouquets = []
        random_bouquets = list(
            map(lambda recipient_id: self._prepare_random_bouquet(flowers_in_market, recipient_id), remaining_receipients))
        for bouquet in random_bouquets:
            # TODO: Do something with these values
            self.controlled_given[bouquet[1]].append([bouquet[2], 0])
        for k, v in chosen_bouquets.items():
            bouquets.append((self.suitor_id, k, Bouquet({v: 1})))
        return bouquets + random_bouquets

    def _prepare_random_bouquet(self, remaining_flowers, recipient_id):
        num_remaining = len(remaining_flowers)
        size = int(np.random.randint(0, min(MAX_BOUQUET_SIZE, num_remaining) + 1))
        if size > 0:
            chosen_flowers = np.random.choice(remaining_flowers, size=(size, ), replace=False)
            chosen_flower_counts = dict(Counter(chosen_flowers))
            for flower in chosen_flowers:
                remaining_flowers.remove(flower)
        else:
            chosen_flower_counts = dict()
        chosen_bouquet = Bouquet(chosen_flower_counts)
        return self.suitor_id, recipient_id, chosen_bouquet
