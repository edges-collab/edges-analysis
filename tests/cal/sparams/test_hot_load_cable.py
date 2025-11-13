from edges.cal import sparams as sp


class TestSparamsFromSemiRigid:
    def test_2017_semi_rigid(self):
        hlc = sp.read_semi_rigid_cable_sparams_file(
            path=":semi_rigid_s_parameters_2017.txt"
        )
        assert hlc.s12.dtype == complex
