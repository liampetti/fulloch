from tests.conftest import _StubModule


def test_stub_module_attributes_are_stable(monkeypatch):
    stub = _StubModule("example")

    assert stub.float32 is stub.float32
    monkeypatch.setattr(stub.cuda, "is_available", lambda: True)
    assert stub.cuda.is_available()
    assert stub.SomeClass() is not stub.SomeClass()
