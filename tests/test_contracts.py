import json
import tempfile
import unittest
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
from phasegraph import DistanceModel, read_graph, write_graph
from phasegraph.analysis import cluster_matrix

def graph(name, cost=.2):
    g = nx.Graph(name=name)
    g.add_edge("a", "b", edge_distance=cost)
    g.add_edge("b", "c", edge_distance=cost*1.5)
    return g

class Contracts(unittest.TestCase):
    def test_isolates_and_disconnected_graph_roundtrip(self):
        g = graph("regions"); g.add_node("isolated", position=[2.,3.])
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"g.json"; write_graph(g,p); h=read_graph(p)
            self.assertEqual(set(g),set(h)); self.assertEqual(h.nodes["isolated"]["position"],[2.,3.])
            self.assertEqual(DistanceModel().fit([g,graph("other")]).matrix().shape,(2,2))

    def test_rename_and_identity(self):
        g=graph("original"); h=nx.relabel_nodes(g,{"a":"z","b":"x","c":"y"}); h.graph["name"]="renamed"
        d=DistanceModel().fit([g,h]).matrix()
        np.testing.assert_allclose(d,0,atol=1e-8)

    def test_gallery_is_frozen_after_queries(self):
        model=DistanceModel().fit([graph("a"),graph("b",.3)])
        before=model.matrix().copy(); bins=model.breaks.copy(); sigma=model.sigma
        model.query(graph("new",.1)); model.query(graph("new2",.25))
        np.testing.assert_array_equal(before,model.matrix()); np.testing.assert_array_equal(bins,model.breaks)
        self.assertEqual(sigma,model.sigma)

    def test_overflow_rejected_and_explicit_clipping_reported(self):
        with self.assertRaisesRegex(ValueError,"exceed frozen"):
            DistanceModel().fit([graph("ref")]).query(graph("large",3))
        ranked=DistanceModel(overflow="clip").fit([graph("ref")]).query(graph("large",3))
        self.assertTrue(ranked[0]["query_overflow"])

    def test_save_reload(self):
        model=DistanceModel().fit([graph("a"),graph("b",.3)])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"model.json"; model.save(p)
            np.testing.assert_array_equal(model.matrix(),DistanceModel.load(p).matrix())
            self.assertEqual(model.query(graph("q",.1)),DistanceModel.load(p).query(graph("q",.1)))

    def test_singletons_and_no_edges(self):
        a=nx.Graph(name="one"); a.add_node("x")
        b=nx.Graph(name="two"); b.add_nodes_from(["a","b"])
        d=DistanceModel().fit([a,b]).matrix()
        self.assertTrue(np.isfinite(d).all()); self.assertTrue(d[0,1]>0)

    def test_invalid_costs_and_duplicate_edges_rejected(self):
        for cost in [0,-.1,float("inf"),float("nan")]:
            with self.assertRaises(ValueError): DistanceModel().fit([graph("bad",cost)])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"edges.csv"; p.write_text("node_u,node_v,edge_distance\na,b,.2\nb,a,.2\n")
            with self.assertRaisesRegex(ValueError,"Duplicate"): read_graph(p)

    def test_malformed_model_and_matrix_fail(self):
        model=DistanceModel().fit([graph("a"),graph("b",.3)])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"m.json"; model.save(p); data=json.loads(p.read_text()); data["features"][0]["mean"]=[float("nan")]; p.write_text(json.dumps(data))
            with self.assertRaises(ValueError): DistanceModel.load(p)
        with self.assertRaises(ValueError): cluster_matrix(pd.DataFrame([[0,1],[2,0]]),2)

    def test_finite_tail_keeps_gallery_and_separates_unreachable(self):
        gallery=[graph('a'),graph('b',.3)]
        strict=DistanceModel().fit(gallery);tail=DistanceModel(overflow='tail').fit(gallery)
        np.testing.assert_allclose(strict.matrix(),tail.matrix(),atol=1e-14)
        before=tail.matrix().copy();long=tail.encode(graph('long',3))
        disconnected=nx.Graph(name='disconnected');disconnected.add_nodes_from(['a','b','c'])
        missing=tail.encode(disconnected)
        self.assertTrue(long['overflow']);self.assertGreater(long['mean'][-2],0)
        self.assertEqual(long['mean'][-1],0);self.assertEqual(missing['mean'][-2],0)
        self.assertEqual(missing['mean'][-1],1);self.assertGreater(tail.compare_features(long,missing),0)
        tail.query(graph('query',3));np.testing.assert_array_equal(before,tail.matrix())
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'tail.json';tail.save(p)
            self.assertEqual(tail.query(graph('q',3)),DistanceModel.load(p).query(graph('q',3)))

    def test_geometry_options_keep_default_and_reject_conflicts(self):
        from phasegraph.geometry import extract_array
        from phasegraph.benchmark import synth_fixture
        image,_,_,_=synth_fixture(48,'clean')
        original,debug=extract_array(image)
        explicit,other=extract_array(image,repair_strategy='local',canonical_width=image.shape[1])
        self.assertEqual(list(original.edges(data=True)),list(explicit.edges(data=True)))
        np.testing.assert_array_equal(debug['labels'],other['labels'])
        for kwargs in [{'repair_strategy':'unknown'},{'repair':False,'repair_strategy':'local'},{'canonical_width':10}]:
            with self.assertRaises(ValueError):extract_array(image,**kwargs)
        dark_blue=np.zeros((200,300,3),dtype=np.uint8); dark_blue[:]=[120,0,0]
        with self.assertRaisesRegex(ValueError,'Unsupported image format'): extract_array(dark_blue)

    def test_mask_matching_detects_region_merging(self):
        from phasegraph.evaluation import match_regions,score_contacts
        reference=np.array([[1,1,2,2],[1,1,2,2]],dtype=np.int32)
        prediction=np.array([[4,4,7,7],[4,4,7,7]],dtype=np.int32)
        mapping,_=match_regions(reference,prediction,[1,2],[4,7])
        self.assertEqual(mapping,{4:1,7:2})
        self.assertEqual(score_contacts([(1,2)],[(4,7)],mapping,2,2)['edge_F1'],1)
        merged=np.full_like(reference,4);mapping,_=match_regions(reference,merged,[1,2],[4])
        score=score_contacts([(1,2)],[],mapping,2,1)
        self.assertFalse(score['exact_graph_recovery']);self.assertEqual(score['edge_F1'],0)

    def test_axis_coordinates_are_invariant_without_changing_contacts(self):
        from phasegraph.geometry import coordinate_cost, extract_array
        from phasegraph.benchmark import synth_fixture
        for sx, sy in [(0.8, 1.2), (2., .5), (1., 1.)]:
            self.assertEqual(coordinate_cost([123*sx, 74*sy], [800*sx, 600*sy], 'axes'),
                             coordinate_cost([123, 74], [800, 600], 'axes'))
        self.assertNotEqual(coordinate_cost([123, 74], [800, 600]), coordinate_cost([246, 74], [1600, 600]))
        image, _, _, _ = synth_fixture(48, 'clean')
        legacy, a = extract_array(image)
        axes, b = extract_array(image, coordinate_metric='axes')
        self.assertEqual(set(legacy.edges), set(axes.edges))
        np.testing.assert_array_equal(a['labels'], b['labels'])
        self.assertEqual(axes.graph['coordinate_metric'], 'axes')
        self.assertEqual(b['config']['coordinate_metric'], 'axes')
        legacy.graph['name'] = 'legacy_diagonal'
        model = DistanceModel().fit([axes])
        with self.assertRaisesRegex(ValueError, 'coordinate_metric'): model.query(legacy)
        with self.assertRaisesRegex(ValueError, 'coordinate_metric'): DistanceModel().fit([axes, legacy])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'axes.json'; model.save(p)
            self.assertEqual(DistanceModel.load(p).coordinate_metric, 'axes')
            self.assertEqual(model.query(axes), DistanceModel.load(p).query(axes))
        with self.assertRaises(ValueError): extract_array(image, coordinate_metric='other')
        with self.assertRaises(ValueError): coordinate_cost([1, 2], [0, 600], 'axes')

    def test_coupled_hypotheses_preserve_shared_score_changes(self):
        from phasegraph.stability import assess_distances
        d=np.array([[.1,.2,.8,.9],[1.1,1.2,1.8,1.9]])
        report=assess_distances(d,['a','b','c','d'],top_k=2)
        self.assertTrue(report['coupled_certified']);self.assertFalse(report['box_certified'])
        self.assertEqual(report['nominal_top_k'],['a','b'])
        self.assertAlmostEqual(report['coupled_margin'],.6)
        crossing=assess_distances([[.1,.2,.8,.9],[.8,.2,.1,.9]],['a','b','c','d'],top_k=2)
        self.assertFalse(crossing['coupled_certified']);self.assertFalse(crossing['observed_sets_identical'])

    def test_certificate_implications_under_random_hypotheses_and_ties(self):
        from phasegraph.stability import assess_distances
        rng=np.random.default_rng(20261018)
        for _ in range(100):
            d=np.maximum(rng.random((4,9))+.2*rng.random((4,1)),0)
            r=assess_distances(d,list('abcdefghi'),top_k=3)
            if r['box_certified']:self.assertTrue(r['coupled_certified'])
            if r['coupled_certified']:
                expected=set(np.argsort(d[0])[:3])
                self.assertTrue(all(set(np.argsort(row)[:3])==expected for row in d))
        tied=assess_distances([[.1,.1,.3],[.1,.1,.3]],['a','b','c'],top_k=1)
        self.assertFalse(tied['coupled_certified'])
        with self.assertRaises(ValueError):assess_distances([[0,float('nan'),.3]],['a','b','c'])

    def test_continuous_histograms_remove_boundary_jumps_and_normalize_mass(self):
        from phasegraph.distance import entropy
        two=lambda cost: nx.Graph([('a','b',{'edge_distance':cost})])
        hard=DistanceModel(bins=2,overflow='tail').fit([graph('calibration')])
        smooth=DistanceModel(bins=2,overflow='tail',encoding='continuous').fit([graph('calibration')])
        hard.breaks=np.array([0.,.5,1.]);smooth.breaks=hard.breaks.copy()
        hard.sigma=smooth.sigma=.2
        a,b=two(.5-1e-5),two(.5+1e-5)
        self.assertGreater(hard.compare_features(hard.encode(a),hard.encode(b)),.4)
        self.assertLess(smooth.compare_features(smooth.encode(a),smooth.encode(b)),1e-3)
        for cost in [.01,.25,.5,.75,1.,1.1,2.]:
            f=smooth.encode(two(cost));self.assertAlmostEqual(sum(f['mean']),1.)
            self.assertGreaterEqual(min(f['mean']),0.)
        near_a=smooth.encode(two(1-1e-6));near_b=smooth.encode(two(1+1e-6))
        self.assertLess(smooth.compare_features(near_a,near_b),.002)

    def test_continuous_models_roundtrip_and_keep_tail_gallery_distances(self):
        a=graph('a');b=graph('b',.3)
        strict=DistanceModel(encoding='continuous').fit([a,b])
        tail=DistanceModel(encoding='continuous',overflow='tail').fit([a,b])
        np.testing.assert_allclose(strict.matrix(),tail.matrix(),atol=1e-12)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'continuous.json';tail.save(p);loaded=DistanceModel.load(p)
            self.assertEqual(loaded.encoding,'continuous')
            self.assertEqual(tail.query(a),loaded.query(a))
        with self.assertRaises(ValueError):DistanceModel(encoding='invalid')

    def test_missing_hypothesis_blocks_certificate_and_gallery_stays_frozen(self):
        from phasegraph.stability import query_hypotheses
        model=DistanceModel().fit([graph('a'),graph('b',.3),graph('c',.4)])
        before=model.matrix().copy()
        report=query_hypotheses(model,[graph('q',.2),graph('impossible',100)],top_k=1)
        self.assertEqual(report['status'],'incomplete');self.assertFalse(report['coupled_certified'])
        self.assertEqual(report['successful_hypotheses'],1)
        np.testing.assert_array_equal(before,model.matrix())

if __name__=="__main__": unittest.main()
