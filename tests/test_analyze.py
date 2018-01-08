# -*- mode: python; coding: utf-8 -*-
#
# Copyright (C) 2017 <contact@redhat.com>
#
# Author: Loic Dachary <loic@dachary.org>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
from multiprocessing import Pool
import copy
import collections
import logging
import json
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import entropy
import pytest # noqa needed for capsys
import pprint

from crush import Crush
from crush.main import Main
from crush.ceph import Ceph
from crush.analyze import Analyze

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG)


def o(a):
    return optimize(*a)

# FAIL if "straw"
def optimize(p, crushmap, bucket, with_positions):
    if len(bucket.get('children', [])) == 0:
        return None
    print("Optimizing " + bucket['name'])
    crushmap = copy.deepcopy(crushmap)
    a = Ceph().constructor([
        'analyze',
    ] + p)
    id2weight = collections.OrderedDict([ (i['id'], i['weight']) for i in bucket['children'] ])

    if with_positions:
        choose_arg = {
            'bucket_id': bucket['id'],
            'weight_set': [
                id2weight.values(),
            ] * a.args.replication_count
        }
        crushmap['choose_args']['optimize'].append(choose_arg)
        for replication_count in range(1, a.args.replication_count + 1):
            print("Improving replica " + str(replication_count))
            pprint.pprint(choose_arg['weight_set'])
            optimize_replica(a, crushmap, bucket, replication_count, choose_arg, replication_count-1)
    else:
        choose_arg = {
            'bucket_id': bucket['id'],
            'weight_set': [
                id2weight.values(),
            ]
        }
        crushmap['choose_args']['optimize'].append(choose_arg)
        optimize_replica(a, crushmap, bucket, a.args.replication_count, choose_arg, 0)

    print(bucket['name'] + " weights " + str(id2weight.values()))
    pprint.pprint(choose_arg['weight_set'])
    return choose_arg
    
def optimize_replica(a, crushmap, bucket, replication_count, choose_arg, choose_arg_position):
    a.args.replication_count = replication_count
    id2weight = collections.OrderedDict([ (i['id'], i['weight']) for i in bucket['children'] ])
    
    c = Crush(backward_compatibility=True)
    c.parse(crushmap)

    (take, failure_domain) = c.rule_get_take_failure_domain(a.args.rule)
    pd.set_option('precision', 2)

    c.parse(crushmap)
    #
    # initial simulation
    #
    i = a.run_simulation(c, take, failure_domain)
    #print(str(i))
    # select desired items
    i = i.reset_index()
    s = i['~name~'] == 'KKKK' # init to False, there must be a better way
    for item in bucket['children']:
        s |= (i['~name~'] == item['name'])
    i['~delta~'] = i.loc[s, '~objects~'] - i.loc[s, '~expected~']
    i.loc[s, '~delta%~'] = (i.loc[s, '~objects~'] - i.loc[s, '~expected~']) / i.loc[s, '~expected~'] * 100
    i = i.sort_values('~delta~', ascending=False)
    i = i[pd.notnull(i['~delta~'])]

    previous_kl = None
    improve_tolerance = 10
    no_improvement = 0
    max_iterations = 1000
    for iterations in range(max_iterations):
#        print(bucket['name'] + " weights " + str(id2weight.values()))
        choose_arg['weight_set'][choose_arg_position] = id2weight.values()
#        a.args.verbose = 1
        c.parse(crushmap)
        z = a.run_simulation(c, take, failure_domain)
        z = z.reset_index()
        d = z[s].copy()
        d['~delta~'] = d['~objects~'] - d['~expected~']
        d['~delta%~'] = d['~delta~'] / d['~expected~']
        kl = d['~delta~'].abs().sum()
        # kl = entropy(d.loc[s, '~expected~'], d.loc[s, '~objects~'])
        # stop when kl is small enough or when it increases meaning
        # what we're doing is no longer reducing kl
        if previous_kl is not None:
            if previous_kl < kl:
                no_improvement += 1
            else:
                previous_kl = kl
                best_weights = id2weight.values()
                no_improvement = 0
            if no_improvement >= improve_tolerance:
                choose_arg['weight_set'][choose_arg_position] = best_weights
                break
        else:
            best_weights = id2weight.values()
            previous_kl = kl
        print(bucket['name'] + " kl " + str(kl) + " no_improvement " + str(no_improvement))
        # if kl < 1e-6:
        #     break
        d = d.sort_values('~delta~', ascending=False)
#        print(str(d))
        if d.iloc[0]['~delta~'] <= 0 or d.iloc[-1]['~delta~'] >= 0:
            break
        # there should not be a need to keep the sum of the weights to the same value, they
        # are only used locally for placement and have no impact on the upper weights
        # nor are they derived from the weights from below *HOWEVER* in case of a failure
        # the weights need to be as close as possible from the target weight to limit
        # the negative impact
        shift = id2weight[d.iloc[0]['~id~']] * min(0.01, d.iloc[0]['~delta%~'])
        if id2weight[d.iloc[-1]['~id~']] < shift:
            break
        id2weight[d.iloc[0]['~id~']] -= shift
        id2weight[d.iloc[-1]['~id~']] += shift
    if iterations >= max_iterations - 1:
        print("!!!!!!!! stoped after " + str(iterations))
    print("Done " + str(no_improvement))


class TestAnalyze(object):

    def test_collect_dataframe(self):
        tree = {
            'name': 'rack0', 'type': 'rack', 'id': -1, 'children': [
                {'name': 'host0', 'type': 'host', 'id': -2, 'children': [
                    {'name': 'osd.3', 'id': 3},
                ]},
                {'name': 'host1', 'type': 'host', 'id': -3, 'children': [
                    {'name': 'osd.4', 'id': 4},
                ]},
            ]
        }
        c = Crush(verbose=1)
        c.parse({"trees": [tree]})
        d = Analyze.collect_dataframe(c, tree)
        expected = """\
        ~id~  ~weight~  ~type~   rack   host device
~name~                                             
rack0     -1       1.0    rack  rack0    NaN    NaN
host0     -2       1.0    host  rack0  host0    NaN
osd.3      3       1.0  device  rack0  host0  osd.3
host1     -3       1.0    host  rack0  host1    NaN
osd.4      4       1.0  device  rack0  host1  osd.4\
""" # noqa trailing whitespaces are expected
        assert expected == str(d)

    # see "Proposal for a CRUSH collision fallback" on ceph-devel
    # for the required bug fix

    # CRUSH weights 5 1 1 1 1 cannot lead to an even distribution for
    # two replicas because the item with weight 5 gets 56% of the
    # values for the first replica. And only 44% of the values for the
    # second replica can be placed on the same item. The 12%
    # difference stays. I spent hours blaming the optimization
    # algorithm for not solving the impossible.
    
    def test_fix_impossible(self):
        # [ 5 1 1 1 1]
        size = 2
        pg_num = 2048
        p = [
            '--replication-count', str(size),
            '--pool', '0',
            '--pg-num', str(pg_num),
            '--pgp-num', str(pg_num),
        ]

        device_count = 0
        hosts_count = 5
        host_weight = [ 1 ] * hosts_count
        host_weight[0] = 5
        crushmap = {
            "trees": [
                {
                    "type": "root",
                    "id": -1,
                    "name": "dc1",
                    "weight": sum(host_weight),
                    "children": [],
                }
            ],
            "rules": {
                "firstn": [
                    ["take", "dc1"],
#                    ["set_choose_tries", 20000],
                    ["choose", "firstn", 0, "type", "host"],
                    ["emit"]
                ],
            }
        }
        crushmap['trees'][0]['children'].extend([
            {
                "type": "host",
                "id": -(i + 2),
                "name": "host%d" % i,
                "weight": host_weight[i],
                "children": [
                    {"id": (device_count * i + d), "name": "device%02d" % (device_count * i + d), "weight": device_weights[d]}
                    for d in range(device_count)],
            } for i in range(0, hosts_count)
        ])
        with pytest.raises(ValueError):
            self.run_optimize(p, 'firstn', crushmap)

    def test_fix_0(self):
        # [ 5 1 1 1 1 1 1 1 1 1 ]
        size = 2
        pg_num = 51200
        p = [
            '--replication-count', str(size),
            '--pool', '0',
            '--pg-num', str(pg_num),
            '--pgp-num', str(pg_num),
        ]

        device_count = 0
        hosts_count = 10
        host_weight = [ 1 ] * hosts_count
        host_weight[0] = 5
        crushmap = {
            "trees": [
                {
                    "type": "root",
                    "id": -1,
                    "name": "dc1",
                    "weight": sum(host_weight),
                    "children": [],
                }
            ],
            "rules": {
                "firstn": [
                    ["take", "dc1"],
                    ["set_choose_tries", 100],
                    ["choose", "firstn", 0, "type", "host"],
                    ["emit"]
                ],
            }
        }
        crushmap['trees'][0]['children'].extend([
            {
                "type": "host",
                "id": -(i + 2),
                "name": "host%d" % i,
                "weight": host_weight[i],
                "children": [
                    {"id": (device_count * i + d), "name": "device%02d" % (device_count * i + d), "weight": device_weights[d]}
                    for d in range(device_count)],
            } for i in range(0, hosts_count)
        ])
        self.run_optimize(p, 'firstn', crushmap)

    def test_fix_1(self):
        # [ 5 1 1 1 1 1 1 1 1 1 ]
        size = 2
        pg_num = 512
        p = [
            '--replication-count', str(size),
            '--pool', '0',
            '--pg-num', str(pg_num),
            '--pgp-num', str(pg_num),
        ]

        device_count = 0
        hosts_count = 10
        host_weight = [ 1 ] * hosts_count
        host_weight[0] = 5
        crushmap = {
            "trees": [
                {
                    "type": "root",
                    "id": -1,
                    "name": "dc1",
                    "weight": sum(host_weight),
                    "children": [],
                }
            ],
            "rules": {
                "firstn": [
                    ["take", "dc1"],
                    ["set_choose_tries", 100],
                    ["choose", "firstn", 0, "type", "host"],
                    ["emit"]
                ],
            }
        }
        crushmap['trees'][0]['children'].extend([
            {
                "type": "host",
                "id": -(i + 2),
                "name": "host%d" % i,
                "weight": host_weight[i],
                "children": [
                    {"id": (device_count * i + d), "name": "device%02d" % (device_count * i + d), "weight": device_weights[d]}
                    for d in range(device_count)],
            } for i in range(0, hosts_count)
        ])
        self.run_optimize(p, 'firstn', crushmap)

    def test_fix_2(self):
        # [ 1 2 3 4 5 6 7 8 9 10 ... 100 ]
        size = 2
        hosts_count = 100
        pg_num = hosts_count * 200
        p = [
            '--replication-count', str(size),
            '--pool', '0',
            '--pg-num', str(pg_num),
            '--pgp-num', str(pg_num),
        ]

        device_count = 0
        host_weight = [ i for i in range(1, hosts_count + 1) ]
        crushmap = {
            "trees": [
                {
                    "type": "root",
                    "id": -1,
                    "name": "dc1",
                    "weight": sum(host_weight),
                    "children": [],
                }
            ],
            "rules": {
                "firstn": [
                    ["take", "dc1"],
                    ["set_choose_tries", 100],
                    ["choose", "firstn", 0, "type", "host"],
                    ["emit"]
                ],
            }
        }
        crushmap['trees'][0]['children'].extend([
            {
                "type": "host",
                "id": -(i + 2),
                "name": "host%d" % i,
                "weight": host_weight[i],
                "children": [
                    {"id": (device_count * i + d), "name": "device%02d" % (device_count * i + d), "weight": device_weights[d]}
                    for d in range(device_count)],
            } for i in range(0, hosts_count)
        ])

        self.run_optimize(p, 'firstn', crushmap)

    def test_fix_3(self):
        # [ 1 2 3 1 2 3 1 2 3 1 ]
        size = 2
        pg_num = 512
        p = [
            '--replication-count', str(size),
            '--pool', '0',
            '--pg-num', str(pg_num),
            '--pgp-num', str(pg_num),
        ]

        device_count = 0
        hosts_count = 10
        host_weight = [ i % 3 + 1 for i in range(hosts_count) ]
        crushmap = {
            "trees": [
                {
                    "type": "root",
                    "id": -1,
                    "name": "dc1",
                    "weight": sum(host_weight),
                    "children": [],
                }
            ],
            "rules": {
                "firstn": [
                    ["take", "dc1"],
                    ["set_choose_tries", 100],
                    ["choose", "firstn", 0, "type", "host"],
                    ["emit"]
                ],
            }
        }
        crushmap['trees'][0]['children'].extend([
            {
                "type": "host",
                "id": -(i + 2),
                "name": "host%d" % i,
                "weight": host_weight[i],
                "children": [
                    {"id": (device_count * i + d), "name": "device%02d" % (device_count * i + d), "weight": device_weights[d]}
                    for d in range(device_count)],
            } for i in range(0, hosts_count)
        ])

        self.run_optimize(p, 'firstn', crushmap)

    def test_fix_5(self):
        p = [
            '--replication-count', str(3),
            '--pool', '2',
            '--pg-num', str(2048),
            '--pgp-num', str(2048),
        ]
        self.run_optimize(p, 'replicated_ruleset', 'tests/test_analyze_fix_5.json')

    def test_fix_6(self):
        p = [
            '--replication-count', str(3),
            '--pool', '2',
            '--pg-num', str(2048),
            '--pgp-num', str(2048),
        ]
        self.run_optimize(p, 'replicated_ruleset', 'tests/test_analyze_fix_6.json')

    def test_fix_8(self):
        p = [
            '--replication-count', str(3),
            '--pool', '5',
            '--pg-num', str(2048),
            '--pgp-num', str(2048),
        ]
        self.run_optimize(p, 'data', 'tests/test_analyze_fix_8.json')

    def run_optimize(self, p, rule_name, crushmap, with_positions=True):
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', 160)
        
        p.extend(['--rule', rule_name])
        a = Ceph().constructor([
            'analyze',
        ] + p)

        c = Crush(backward_compatibility=True)
        c.parse(crushmap)
        (take, failure_domain) = c.rule_get_take_failure_domain(rule_name)
        crushmap = c.get_crushmap()
        crushmap['choose_args'] = { "optimize": [], }

        d = a.run_simulation(c, take, failure_domain)
        if d['~overweight~'].any():
            raise ValueError('no way to optimize when there is an overweight item')
        print(str(d))
        print(a._format_report(d, 'device'))
        print(a._format_report(d, failure_domain))
        print(a.analyze_failures(c, take, failure_domain))

        p.extend(['--choose-args', 'optimize'])

        pool = Pool()
        children = [c.find_bucket(take)]
        while len(children) > 0:
            a = [(p, crushmap, item, with_positions) for item in children]
            r = pool.map(o, a)
#            r = map(o, a)
            choose_args = filter(None, r)
            crushmap['choose_args']['optimize'].extend(choose_args)
            nc = []
            for item in children:
                nc.extend(item.get('children', []))
            # fail if all children are not of the same type
            children = nc

        pprint.pprint(crushmap)
        c.parse(crushmap)
        a = Ceph().constructor([
            'analyze',
        ] + p)
        d = a.run_simulation(c, take, failure_domain)
        print(a._format_report(d, 'device'))
        print(a._format_report(d, failure_domain))
        print(a.analyze_failures(c, take, failure_domain))

    def test_analyze_out_of_bounds(self):
        # [ 5 1 1 1 1]
        size = 2
        pg_num = 2048
        p = [
            '--replication-count', str(size),
            '--pool', '0',
            '--pg-num', str(pg_num),
            '--pgp-num', str(pg_num),
        ]

        hosts_count = 5
        host_weight = [1] * hosts_count
        host_weight[0] = 5
        crushmap = {
            "trees": [
                {
                    "type": "root",
                    "id": -1,
                    "name": "dc1",
                    "weight": sum(host_weight),
                    "children": [],
                }
            ],
            "rules": {
                "firstn": [
                    ["take", "dc1"],
                    ["choose", "firstn", 0, "type", "host"],
                    ["emit"]
                ],
            }
        }
        crushmap['trees'][0]['children'].extend([
            {
                "type": "host",
                "id": -(i + 2),
                "name": "host%d" % i,
                "weight": host_weight[i],
                "children": [],
            } for i in range(0, hosts_count)
        ])
        a = Ceph().constructor([
            'analyze',
            '--rule', 'firstn',
        ] + p)
        a.args.crushmap = crushmap
        d = a.analyze()
        expected = """\
        ~id~  ~weight~  ~objects~  ~over/under used %~
~name~                                                
host3     -5         1        646                41.94
host4     -6         1        610                34.03
host2     -4         1        575                26.34
host1     -3         1        571                25.46
host0     -2         5       1694               -25.56

Worst case scenario if a host fails:

        ~over used %~
~type~               
host            61.52
root             0.00

The following are overweight:

        ~id~  ~weight~
~name~                
host0     -2         5\
""" # noqa trailing whitespaces are expected
        assert expected == str(d)

    def test_analyze(self):
        trees = [
            {"name": "dc1", "type": "root", "id": -1, 'children': []},
        ]
        weights = (
            (10.0, 1.0, 5.0, 4.0),
            (10.0, 1.0, 5.0, 4.0),
            (10.0, 1.0, 5.0, 4.0),
            (10.0, 1.0, 5.0, 4.0),
            (1.0, 0.1, 0.5, 0.4),
        )
        trees[0]['children'].extend([
            {
                "type": "host",
                "id": -(i + 3),
                "name": "host%d" % i,
                "weight": weights[i][0],
                "children": [
                    {"id": (3 * i),
                     "name": "device%02d" % (3 * i), "weight": weights[i][1]},
                    {"id": (3 * i + 1),
                     "name": "device%02d" % (3 * i + 1), "weight": weights[i][2]},
                    {"id": (3 * i + 2),
                     "name": "device%02d" % (3 * i + 2), "weight": weights[i][3]},
                ],
            } for i in range(5)
        ])
        a = Main().constructor([
            'analyze',
            '--rule', 'data',
            '--replication-count', '2',
            '--values-count', '10000',
        ])
        a.args.crushmap = {
            "trees": trees,
            "rules": {
                "data": [
                    ["take", "dc1"],
                    ["chooseleaf", "firstn", 0, "type", "host"],
                    ["emit"]
                ]
            }
        }
        d = a.analyze()
        expected = """\
        ~id~  ~weight~  ~objects~  ~over/under used %~
~name~                                                
host4     -7       1.0        541                10.91
host3     -6      10.0       4930                 1.07
host2     -5      10.0       4860                -0.37
host1     -4      10.0       4836                -0.86
host0     -3      10.0       4833                -0.92

Worst case scenario if a host fails:

        ~over used %~
~type~               
device          25.55
host            22.45
root             0.00\
""" # noqa trailing whitespaces are expected
        assert expected == str(d)

    def test_analyze_weights(self):
        a = Main().constructor(
            ["analyze", "--rule", "replicated_ruleset",
             "--replication-count", "2", "--type", "device",
             "--crushmap", "tests/ceph/dump.json",
             "--weights", "tests/ceph/weights.json"])
        a.args.backward_compatibility = True
        res = a.run()
        assert "-100.00" in str(res)  # One of the OSDs has a weight of 0.0

# Local Variables:
# compile-command: "cd .. ; tox -e py27 -- -s -vv tests/test_analyze.py"
# End:
