// Copyright 2020 The Jujutsu Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Helps build a new `MergedTree` from a base tree and overrides.

use std::collections::BTreeMap;

use itertools::EitherOrBoth;
use itertools::Itertools as _;
use thiserror::Error;

use crate::backend::BackendResult;
use crate::backend::MergedTreeValue;
use crate::backend::TreeId;
use crate::backend::TreeValue;
use crate::merge::Merge;
use crate::merged_tree::MergedTree;
use crate::repo_path::RepoPathBuf;
use crate::repo_path::RepoPathComponentBuf;
use crate::repo_path::RepoPathTree;
use crate::tree::Tree;
use crate::tree_rewriter::RewrittenEntry;
use crate::tree_rewriter::TreeRewriteVisitor;
use crate::tree_rewriter::rewrite_trees;

/// Helper for writing trees with conflicts.
///
/// You start by creating an instance of this type with one or more
/// base trees. You then add overrides on top. The overrides may be
/// conflicts. Then you can write the result as a merge of trees.
#[derive(Debug)]
pub struct MergedTreeBuilder {
    base_tree: MergedTree,
    overrides: BTreeMap<RepoPathBuf, MergedTreeValue>,
}

/// Error returned when attempting to set an override with a conflict arity that
/// does not match the base tree.
#[derive(Debug, Error)]
#[error("Conflict arity mismatch")]
pub struct ConflictArityMismatchError;

impl MergedTreeBuilder {
    /// Create a new builder with the given trees as base.
    pub fn new(base_tree: MergedTree) -> Self {
        Self {
            base_tree,
            overrides: BTreeMap::new(),
        }
    }

    /// Set an override compared to the base tree. Use `None` to remove a
    /// value from the tree.
    pub fn set_or_remove(&mut self, path: RepoPathBuf, value: Option<TreeValue>) {
        self.overrides.insert(path, Merge::resolved(value));
    }

    /// Set an override compared to the base tree. The `values` merge must
    /// either be resolved (i.e. have 1 side) or have the same number of
    /// sides as the base tree used to construct this builder. Conflicted
    /// values should also be derived from the base tree's conflict. To
    /// combine paths from a differently conflicted tree, merge the trees
    /// instead (e.g. using `MergedTree::filtered()` and
    /// `MergedTree::merge()`). Use `Merge::absent()` to remove a value from
    /// the tree.
    pub fn set_or_remove_merge(
        &mut self,
        path: RepoPathBuf,
        values: MergedTreeValue,
    ) -> Result<(), ConflictArityMismatchError> {
        if !(values.is_resolved() || values.num_sides() == self.base_tree.tree_ids().num_sides()) {
            return Err(ConflictArityMismatchError);
        }
        self.overrides.insert(path, values);
        Ok(())
    }

    /// Create new tree(s) from the base tree(s) and overrides.
    pub async fn write_tree(self) -> BackendResult<MergedTree> {
        let store = self.base_tree.store().clone();
        let labels = self.base_tree.labels().clone();
        let new_tree_ids = self.write_merged_trees().await?;
        let (labels, new_tree_ids) = labels.simplify_with(&new_tree_ids);
        match new_tree_ids.into_resolved() {
            Ok(single_tree_id) => Ok(MergedTree::resolved(store, single_tree_id)),
            Err(tree_ids) => {
                let tree = MergedTree::new(store, tree_ids, labels);
                tree.resolve().await
            }
        }
    }

    async fn write_merged_trees(self) -> BackendResult<Merge<TreeId>> {
        let store = self.base_tree.store().clone();
        let base_tree_ids = self.base_tree.into_tree_ids();
        if self.overrides.is_empty() {
            return Ok(base_tree_ids);
        }
        let visitor = OverridesVisitor::new(self.overrides);
        let root_has_conflict = visitor.overrides.value().has_conflict;
        let roots = Merge::from_vec(
            base_tree_ids
                .into_iter()
                .enumerate()
                .map(|(term_index, tree_id)| (tree_id, root_has_conflict.then_some(term_index)))
                .collect_vec(),
        );
        rewrite_trees(&store, roots, &visitor).await
    }
}

#[derive(Debug, Default)]
struct OverrideNode {
    /// The override at this path, if any.
    value: Option<MergedTreeValue>,
    /// Whether there's a conflicted override at or below this path.
    has_conflict: bool,
}

/// Applies overrides to each term of a merged tree. The context is the index
/// of the term if there are conflicted overrides in the tree, or `None` if all
/// overrides in the tree are resolved and therefore the same for all terms.
struct OverridesVisitor {
    overrides: RepoPathTree<OverrideNode>,
}

impl OverridesVisitor {
    fn new(overrides: BTreeMap<RepoPathBuf, MergedTreeValue>) -> Self {
        let mut tree: RepoPathTree<OverrideNode> = RepoPathTree::default();
        for (path, value) in overrides {
            assert!(!path.is_root());
            if !value.is_resolved() {
                for ancestor in path.ancestors() {
                    tree.add(ancestor).value_mut().has_conflict = true;
                }
            }
            tree.add(&path).value_mut().value = Some(value);
        }
        Self { overrides: tree }
    }
}

impl TreeRewriteVisitor for OverridesVisitor {
    type Context = Option<usize>;

    fn visit_tree(
        &self,
        tree: &Tree,
        term_index: &Option<usize>,
    ) -> Vec<(RepoPathComponentBuf, RewrittenEntry<Option<usize>>)> {
        let node = self.overrides.get(tree.dir()).unwrap();
        let children = node.children().sorted_by_key(|(name, _)| *name);
        let mut new_entries = vec![];
        for item in tree
            .data()
            .entries()
            .merge_join_by(children, |entry, (name, _)| entry.name().cmp(name))
        {
            let (name, base_value, child) = match item {
                EitherOrBoth::Left(entry) => {
                    // No overrides at or below this path
                    new_entries.push((
                        entry.name().to_owned(),
                        RewrittenEntry::Value(entry.value().clone()),
                    ));
                    continue;
                }
                EitherOrBoth::Right((name, child)) => (name, None, child),
                EitherOrBoth::Both(entry, (name, child)) => (name, Some(entry.value()), child),
            };
            let override_node = child.value();
            let value = match &override_node.value {
                Some(values) => match values.as_resolved() {
                    Some(value) => value.clone(),
                    None => values.as_slice()[term_index.unwrap()].clone(),
                },
                None => base_value.cloned(),
            };
            if child.has_children() {
                // Overrides below this path are applied to the base tree, even if
                // there's also an override at this path. If the result is empty,
                // the override at this path is used instead (unless it's a tree).
                let base_tree_id = match base_value {
                    Some(TreeValue::Tree(tree_id)) => tree_id.clone(),
                    _ => tree.store().empty_tree_id().clone(),
                };
                let if_empty = value.filter(|value| !matches!(value, TreeValue::Tree(_)));
                new_entries.push((
                    name.to_owned(),
                    RewrittenEntry::Subtree {
                        base_tree_id,
                        context: term_index.filter(|_| override_node.has_conflict),
                        if_empty,
                    },
                ));
            } else if let Some(value) = value {
                new_entries.push((name.to_owned(), RewrittenEntry::Value(value)));
            }
        }
        new_entries
    }
}
