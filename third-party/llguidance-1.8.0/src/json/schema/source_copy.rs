//! Copies use their reached schema topology and the original compiler account.
use super::*;
use derivre::{
    prepared_funding::{Frame, PreparedFunding, Scope},
    ParserAllocationFunding,
};

struct Copy<'a> {
    funding: &'a ParserAllocationFunding,
    frames: &'a Scope<'a, ParserAllocationFunding>,
}

impl Schema {
    pub(in crate::json) fn copy_with_funding(
        &self,
        funding: &ParserAllocationFunding,
    ) -> Result<Self> {
        let scope = Scope::new(funding).map_err(|error| frames::failure(error, funding))?;
        let _frame = scope
            .frame(std::mem::size_of::<(
                &Self,
                &ParserAllocationFunding,
                Copy<'_>,
                Result<Self>,
            )>())
            .map_err(|error| frames::failure(error, funding))?;
        Copy {
            funding,
            frames: &scope,
        }
        .schema(self)
    }
}
impl StringSchema {
    pub(crate) fn copy_with_funding(&self, funding: &ParserAllocationFunding) -> Result<Self> {
        let scope = Scope::new(funding).map_err(|error| frames::failure(error, funding))?;
        let _frame = scope
            .frame(std::mem::size_of::<(
                &Self,
                &ParserAllocationFunding,
                Copy<'_>,
                Result<Self>,
            )>())
            .map_err(|error| frames::failure(error, funding))?;
        Copy {
            funding,
            frames: &scope,
        }
        .string(self)
    }
}

impl Copy<'_> {
    fn frame<T>(&self) -> Result<Frame<'_>> {
        if let Some(error) = self.funding.failure() {
            return Err(error.into());
        }
        let bytes = std::mem::size_of::<T>()
            .checked_add(std::mem::size_of::<(&Self, usize, Result<Frame<'_>>)>())
            .ok_or_else(|| self.funding.storage_overflow())?;
        self.frames
            .frame(bytes)
            .map_err(|error| frames::failure(error, self.funding))
    }

    fn schema(&self, source: &Schema) -> Result<Schema> {
        let _frame = self.frame::<(
            &Self,
            &Schema,
            &ParserAllocationFunding,
            Schema,
            Result<Schema>,
            &str,
            &ArraySchema,
            &ObjectSchema,
            &StringSchema,
            &Vec<Schema>,
        )>()?;
        Ok(match source {
            Schema::Any => Schema::Any,
            Schema::Null => Schema::Null,
            Schema::Boolean(value) => Schema::Boolean(*value),
            Schema::Number(value) => Schema::Number(value.clone()),
            Schema::Unsatisfiable(value) => Schema::Unsatisfiable(value),
            Schema::Ref(value) => Schema::Ref(self.funding.try_copy_str(value)?),
            Schema::String(value) => Schema::String(self.string(value)?),
            Schema::Array(value) => Schema::Array(self.array(value)?),
            Schema::Object(value) => Schema::Object(self.object(value)?),
            Schema::AnyOf(values) => Schema::AnyOf(self.list(values)?),
            Schema::OneOf(values) => Schema::OneOf(self.list(values)?),
        })
    }

    fn array(&self, source: &ArraySchema) -> Result<ArraySchema> {
        let _frame = self.frame::<(
            &Self,
            &ArraySchema,
            Vec<Schema>,
            Option<Box<Schema>>,
            ArraySchema,
            Result<ArraySchema>,
        )>()?;
        Ok(ArraySchema {
            min_items: source.min_items,
            max_items: source.max_items,
            prefix_items: self.list(&source.prefix_items)?,
            items: self.optional(&source.items)?,
        })
    }

    fn object(&self, source: &ObjectSchema) -> Result<ObjectSchema> {
        let _frame = self.frame::<(
            &Self,
            &ObjectSchema,
            ObjectSchema,
            Result<ObjectSchema>,
            IndexMap<String, Schema>,
            IndexMap<String, Schema>,
            IndexSet<String>,
            indexmap::map::Iter<'_, String, Schema>,
            indexmap::set::Iter<'_, String>,
            &String,
            &Schema,
            String,
            Schema,
            Option<Box<Schema>>,
        )>()?;
        let mut properties = IndexMap::new();
        for (name, schema) in &source.properties {
            self.funding.try_insert_index_map(
                &mut properties,
                self.funding.try_copy_str(name)?,
                self.schema(schema)?,
            )?;
        }
        let mut pattern_properties = IndexMap::new();
        for (name, schema) in &source.pattern_properties {
            self.funding.try_insert_index_map(
                &mut pattern_properties,
                self.funding.try_copy_str(name)?,
                self.schema(schema)?,
            )?;
        }
        let mut required = IndexSet::new();
        for name in &source.required {
            self.funding
                .try_insert_index_set(&mut required, self.funding.try_copy_str(name)?)?;
        }
        Ok(ObjectSchema {
            properties,
            pattern_properties,
            required,
            additional_properties: self.optional(&source.additional_properties)?,
            min_properties: source.min_properties,
            max_properties: source.max_properties,
        })
    }

    fn string(&self, source: &StringSchema) -> Result<StringSchema> {
        let _frame = self.frame::<(
            &Self,
            &StringSchema,
            &ParserAllocationFunding,
            Option<RegexAst>,
            StringSchema,
            Result<StringSchema>,
            derivre::RegexAstCopyPlan<'_>,
            Result<derivre::RegexAstCopyPlan<'_>, derivre::RegexAstCopyFailure>,
        )>()?;
        let regex = if let Some(regex) = &source.regex {
            let plan = regex
                .source_copy_plan(self.frames)
                .map_err(|error| derivre::ParserError::cause(error, self.funding))?;
            self.funding.reserve(plan.requirements().required_bytes())?;
            Some(
                plan.compile()
                    .map_err(|error| derivre::ParserError::cause(error, self.funding))?,
            )
        } else {
            None
        };
        Ok(StringSchema {
            min_length: source.min_length,
            max_length: source.max_length,
            regex,
        })
    }

    fn list(&self, values: &[Schema]) -> Result<Vec<Schema>> {
        let _frame = self.frame::<(
            &Self,
            &[Schema],
            Vec<Schema>,
            Result<Vec<Schema>>,
            std::slice::Iter<'_, Schema>,
            &Schema,
            Schema,
        )>()?;
        let mut result = Vec::new();
        for value in values {
            self.funding.try_push(&mut result, self.schema(value)?)?;
        }
        Ok(result)
    }

    fn optional(&self, value: &Option<Box<Schema>>) -> Result<Option<Box<Schema>>> {
        let _frame = self.frame::<(
            &Self,
            &Option<Box<Schema>>,
            &Schema,
            Schema,
            Option<Box<Schema>>,
            Result<Option<Box<Schema>>>,
        )>()?;
        match value {
            Some(value) => Ok(Some(self.funding.try_box(self.schema(value)?)?)),
            None => Ok(None),
        }
    }
}
