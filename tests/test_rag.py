import unittest

from app.rag_app import Chunk, answer, load_chunks, retrieve, tokenize


class RagTests(unittest.TestCase):
    def test_japanese_tokenizer_produces_bigrams(self):
        self.assertIn("水や", tokenize("野菜の水やり"))

    def test_retrieval_ranks_matching_source(self):
        hits = retrieve("野菜の水やり方法", load_chunks())
        self.assertTrue(hits)
        self.assertEqual("garden-watering.md", hits[0]["source"])

    def test_answer_includes_source_citation(self):
        result = answer("堆肥に使える材料", load_chunks())
        self.assertFalse(result["abstained"])
        self.assertIn("composting-guide.md", result["answer"])
        self.assertTrue(result["sources"])

    def test_unrelated_question_abstains(self):
        result = answer("火星探査機の着陸地点", load_chunks())
        self.assertTrue(result["abstained"])
        self.assertEqual([], result["sources"])

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            answer("   ", [Chunk("x.md", "sample")])


if __name__ == "__main__":
    unittest.main()

